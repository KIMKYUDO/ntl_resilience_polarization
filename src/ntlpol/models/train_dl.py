from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ntlpol.evaluation.metrics import evaluate_prediction_frame
from ntlpol.io_utils import read_json, write_table
from ntlpol.logging_utils import setup_logger
from ntlpol.models.datasets import NTLSequenceTabularDataset, TARGET_COLUMNS, load_modeling_arrays
from ntlpol.models.fusion_multitask import MultiTaskSequenceFusionModel
from ntlpol.models.losses import multitask_loss

LOGGER = setup_logger("ntlpol.train_dl")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _predict(model, loader, device) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            out = model(batch["x_seq"], batch["x_tab"])
            rows.append(
                pd.DataFrame(
                    {
                        "sample_id": batch["sample_id"].cpu().numpy(),
                        "pred_delayed_prob": torch.sigmoid(out["delayed_logit"]).cpu().numpy(),
                        "pred_recovery_percentile": out["percentile"].cpu().numpy(),
                        "pred_no_recovery_12m_prob": torch.sigmoid(out["no_recovery_12m_logit"]).cpu().numpy(),
                        "pred_no_recovery_24m_prob": torch.sigmoid(out["no_recovery_24m_logit"]).cpu().numpy(),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _train_one_fold(
    *,
    encoder_type: str,
    x_seq: np.ndarray,
    x_tab: pd.DataFrame,
    y: pd.DataFrame,
    train_ids: list[int],
    test_ids: list[int],
    feature_cols: list[str],
    cfg: dict,
    model_path: Path,
) -> pd.DataFrame:
    device = _device()
    train_ds = NTLSequenceTabularDataset(
        x_seq=x_seq, x_tab=x_tab, y=y, sample_ids=train_ids, feature_cols=feature_cols
    )
    test_ds = NTLSequenceTabularDataset(
        x_seq=x_seq, x_tab=x_tab, y=y, sample_ids=test_ids, feature_cols=feature_cols
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = MultiTaskSequenceFusionModel(
        seq_channels=x_seq.shape[-1],
        tab_features=len(feature_cols),
        encoder_type=encoder_type,
        seq_hidden=cfg.get("seq_hidden", 64),
        tab_hidden=cfg.get("tab_hidden", 64),
        fusion_hidden=cfg.get("fusion_hidden", 128),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    best_loss = float("inf")
    best_state = None
    patience = int(cfg["early_stopping_patience"])
    bad_epochs = 0

    for _epoch in range(int(cfg["max_epochs"])):
        model.train()
        losses = []
        for batch in train_loader:
            batch = _batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            out = model(batch["x_seq"], batch["x_tab"])
            loss, _ = multitask_loss(
                out,
                batch,
                alpha=cfg["loss_weights"]["percentile_reg"],
                beta12=cfg["loss_weights"]["no_recovery_12m_cls"],
                beta24=cfg["loss_weights"]["no_recovery_24m_cls"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else float("inf")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feature_cols": feature_cols, "cfg": cfg}, model_path)
    return _predict(model, test_loader, device)


def train_sequence_model_loo(
    *,
    processed_dir: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    encoder_type: str,
    early_post_months: int = 3,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    cfg = cfg or {}
    train_cfg = {
        "batch_size": int(cfg.get("batch_size", 256)),
        "max_epochs": int(cfg.get("max_epochs", 50)),
        "early_stopping_patience": int(cfg.get("early_stopping_patience", 8)),
        "learning_rate": float(cfg.get("learning_rate", 1e-3)),
        "weight_decay": float(cfg.get("weight_decay", 1e-4)),
        "loss_weights": cfg.get(
            "loss_weights",
            {"percentile_reg": 0.5, "no_recovery_12m_cls": 0.35, "no_recovery_24m_cls": 0.35},
        ),
        "seq_hidden": int(cfg.get("seq_hidden", 64)),
        "tab_hidden": int(cfg.get("tab_hidden", 64)),
        "fusion_hidden": int(cfg.get("fusion_hidden", 128)),
        "dropout": float(cfg.get("dropout", 0.1)),
    }
    x_seq, x_tab, y, index = load_modeling_arrays(
        processed_dir=processed_dir, early_post_months=early_post_months
    )
    splits = read_json(split_path)
    feature_cols = [c for c in x_tab.columns if c not in {"sample_id", "event_id", "grid_id"}]

    if index.empty or x_tab.empty or y.empty or not splits:
        pred = pd.DataFrame()
        metrics = pd.DataFrame()
        write_table(pred, output_dir / f"predictions/{encoder_type}_early{early_post_months}_predictions.parquet", index=False)
        write_table(metrics, output_dir / f"metrics/{encoder_type}_early{early_post_months}_event_metrics.parquet", index=False)
        LOGGER.warning("Empty data/splits. Wrote empty %s outputs.", encoder_type)
        return pred, metrics

    all_preds = []
    rows = []
    y_meta = y[["sample_id", "event_id", "grid_id"] + TARGET_COLUMNS]
    for fold, split in splits.items():
        train_ids = [int(i) for i in split.get("train", [])]
        test_ids = [int(i) for i in split.get("test", [])]
        if not train_ids or not test_ids:
            continue
        pred = _train_one_fold(
            encoder_type=encoder_type,
            x_seq=x_seq,
            x_tab=x_tab,
            y=y,
            train_ids=train_ids,
            test_ids=test_ids,
            feature_cols=feature_cols,
            cfg=train_cfg,
            model_path=output_dir / f"models/{encoder_type}/{encoder_type}_{fold}_early{early_post_months}.pt",
        )
        pred = pred.merge(y_meta, on="sample_id", how="left")
        pred["fold"] = fold
        all_preds.append(pred)
        rows.append({"fold": fold, **evaluate_prediction_frame(pred)})

    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    metrics_df = pd.DataFrame(rows)
    write_table(pred_df, output_dir / f"predictions/{encoder_type}_early{early_post_months}_predictions.parquet", index=False)
    write_table(metrics_df, output_dir / f"metrics/{encoder_type}_early{early_post_months}_event_metrics.parquet", index=False)
    LOGGER.info("Wrote %s predictions for %s samples", encoder_type, len(pred_df))
    return pred_df, metrics_df
