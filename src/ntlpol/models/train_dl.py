from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ntlpol.evaluation.metrics import evaluate_prediction_frame
from ntlpol.io_utils import read_json, write_table
from ntlpol.logging_utils import setup_logger
from ntlpol.models.datasets import TARGET_COLUMNS, load_modeling_arrays
from ntlpol.models.fusion_multitask import MultiTaskSequenceFusionModel
from ntlpol.models.losses import multitask_loss

LOGGER = setup_logger("ntlpol.train_dl")


class FastDictTensorDataset(Dataset):
    """Pre-materialized tensor dataset for faster GPU training."""

    def __init__(
        self,
        *,
        sample_ids: np.ndarray,
        x_seq_full: np.ndarray,
        x_tab_full: np.ndarray,
        y_full: np.ndarray,
        target_cols: list[str],
    ) -> None:
        sample_ids = np.asarray(sample_ids, dtype=np.int64)
        self.sample_id = torch.as_tensor(sample_ids, dtype=torch.long)

        seq = x_seq_full[sample_ids].astype(np.float32, copy=False)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        self.x_seq = torch.from_numpy(np.ascontiguousarray(seq))

        tab = x_tab_full[sample_ids].astype(np.float32, copy=False)
        tab = np.nan_to_num(tab, nan=0.0, posinf=0.0, neginf=0.0)
        self.x_tab = torch.from_numpy(np.ascontiguousarray(tab))

        yy = y_full[sample_ids].astype(np.float32, copy=False)
        self.y = torch.from_numpy(np.ascontiguousarray(yy))
        self.target_cols = list(target_cols)

    def __len__(self) -> int:
        return int(self.sample_id.numel())

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        out = {
            "sample_id": self.sample_id[idx],
            "x_seq": self.x_seq[idx],
            "x_tab": self.x_tab[idx],
        }

        # Original target columns from y_multitask.parquet:
        #   0: y_delayed_slowest_20pct
        #   1: recovery_delay_percentile
        #   2: y_no_recovery_12m
        #   3: y_no_recovery_24m
        #
        # multitask_loss() expects shorter alias keys:
        #   y_delayed, y_percentile, y_no_recovery_12m, y_no_recovery_24m
        for j, col in enumerate(self.target_cols):
            out[col] = self.y[idx, j]

        out["y_delayed"] = self.y[idx, 0]
        out["y_percentile"] = self.y[idx, 1]
        out["recovery_delay_percentile"] = self.y[idx, 1]

        out["y_no_recovery_12m"] = self.y[idx, 2]
        out["y_no_recovery_24m"] = self.y[idx, 3]

        # aliases expected by multitask_loss()
        out["y_no12"] = self.y[idx, 2]
        out["y_no24"] = self.y[idx, 3]
        return out


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def _make_loader(
    ds: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def _predict(model, loader, device, *, amp: bool, fold: str) -> pd.DataFrame:
    rows = []
    model.eval()
    t0 = time.time()
    total_batches = len(loader)

    with torch.no_grad():
        for bidx, batch in enumerate(loader, start=1):
            batch = _batch_to_device(batch, device)
            with torch.amp.autocast(device_type="cuda", enabled=amp):
                out = model(batch["x_seq"], batch["x_tab"])

            rows.append(
                pd.DataFrame(
                    {
                        "sample_id": batch["sample_id"].detach().cpu().numpy(),
                        "pred_delayed_prob": torch.sigmoid(out["delayed_logit"]).detach().float().cpu().numpy(),
                        "pred_recovery_percentile": out["percentile"].detach().float().cpu().numpy(),
                        "pred_no_recovery_12m_prob": torch.sigmoid(out["no_recovery_12m_logit"]).detach().float().cpu().numpy(),
                        "pred_no_recovery_24m_prob": torch.sigmoid(out["no_recovery_24m_logit"]).detach().float().cpu().numpy(),
                    }
                )
            )

            if bidx == 1 or bidx % 25 == 0 or bidx == total_batches:
                LOGGER.info(
                    "[%s] predict batch %s/%s elapsed=%.1fs",
                    fold,
                    bidx,
                    total_batches,
                    time.time() - t0,
                )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _train_one_fold(
    *,
    encoder_type: str,
    x_seq: np.ndarray,
    x_tab_values: np.ndarray,
    y_values: np.ndarray,
    target_cols: list[str],
    train_ids: list[int],
    test_ids: list[int],
    feature_cols: list[str],
    cfg: dict,
    model_path: Path,
    fold: str,
) -> pd.DataFrame:
    device = _device()
    use_cuda = device.type == "cuda"
    amp = bool(cfg.get("amp", True)) and use_cuda

    train_ids_np = np.asarray(train_ids, dtype=np.int64)
    test_ids_np = np.asarray(test_ids, dtype=np.int64)

    LOGGER.info(
        "[%s] Preparing tensors: train=%s test=%s seq_shape=%s tab_features=%s device=%s amp=%s",
        fold,
        len(train_ids_np),
        len(test_ids_np),
        tuple(x_seq.shape),
        len(feature_cols),
        device,
        amp,
    )

    t_prep = time.time()
    train_ds = FastDictTensorDataset(
        sample_ids=train_ids_np,
        x_seq_full=x_seq,
        x_tab_full=x_tab_values,
        y_full=y_values,
        target_cols=target_cols,
    )
    test_ds = FastDictTensorDataset(
        sample_ids=test_ids_np,
        x_seq_full=x_seq,
        x_tab_full=x_tab_values,
        y_full=y_values,
        target_cols=target_cols,
    )
    LOGGER.info("[%s] Tensor preparation done in %.1fs", fold, time.time() - t_prep)

    train_loader = _make_loader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=use_cuda,
    )
    test_loader = _make_loader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=use_cuda,
    )

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
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_loss = float("inf")
    best_state = None
    patience = int(cfg["early_stopping_patience"])
    bad_epochs = 0
    max_epochs = int(cfg["max_epochs"])

    LOGGER.info(
        "[%s] Start training: epochs=%s patience=%s batch_size=%s train_batches=%s test_batches=%s",
        fold,
        max_epochs,
        patience,
        cfg["batch_size"],
        len(train_loader),
        len(test_loader),
    )

    fold_t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        epoch_t0 = time.time()
        total_batches = len(train_loader)

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        for bidx, batch in enumerate(train_loader, start=1):
            batch = _batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=amp):
                out = model(batch["x_seq"], batch["x_tab"])
                loss, _ = multitask_loss(
                    out,
                    batch,
                    alpha=cfg["loss_weights"]["percentile_reg"],
                    beta12=cfg["loss_weights"]["no_recovery_12m_cls"],
                    beta24=cfg["loss_weights"]["no_recovery_24m_cls"],
                )

            if not torch.isfinite(loss):
                LOGGER.warning(
                    "[%s] epoch %s/%s batch %s/%s non-finite loss=%s; skipping batch",
                    fold,
                    epoch,
                    max_epochs,
                    bidx,
                    total_batches,
                    loss.detach().float().cpu().item() if loss.numel() == 1 else loss,
                )
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()

            loss_val = float(loss.detach().float().cpu())
            losses.append(loss_val)

            if bidx == 1 or bidx % int(cfg["log_every_batches"]) == 0 or bidx == total_batches:
                if use_cuda:
                    mem_gb = torch.cuda.max_memory_allocated() / 1024**3
                    LOGGER.info(
                        "[%s] epoch %s/%s batch %s/%s loss=%.5f gpu_mem=%.2fGB elapsed=%.1fs",
                        fold,
                        epoch,
                        max_epochs,
                        bidx,
                        total_batches,
                        loss_val,
                        mem_gb,
                        time.time() - epoch_t0,
                    )
                else:
                    LOGGER.info(
                        "[%s] epoch %s/%s batch %s/%s loss=%.5f elapsed=%.1fs",
                        fold,
                        epoch,
                        max_epochs,
                        bidx,
                        total_batches,
                        loss_val,
                        time.time() - epoch_t0,
                    )

        epoch_loss = float(np.mean(losses)) if losses else float("inf")
        improved = epoch_loss < best_loss

        if improved:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        LOGGER.info(
            "[%s] epoch %s/%s done train_loss=%.6f best=%.6f improved=%s bad_epochs=%s/%s epoch_time=%.1fs total=%.1fs",
            fold,
            epoch,
            max_epochs,
            epoch_loss,
            best_loss,
            improved,
            bad_epochs,
            patience,
            time.time() - epoch_t0,
            time.time() - fold_t0,
        )

        if bad_epochs >= patience:
            LOGGER.info("[%s] Early stopping at epoch %s", fold, epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "feature_cols": feature_cols, "cfg": cfg},
        model_path,
    )
    LOGGER.info("[%s] Saved model: %s", fold, model_path)

    return _predict(model, test_loader, device, amp=amp, fold=fold)


def _prepare_tabular_matrix(x_tab: pd.DataFrame, feature_cols: list[str], n_samples: int) -> np.ndarray:
    tab = x_tab.set_index("sample_id").reindex(np.arange(n_samples))
    tab_df = tab[feature_cols].astype("float32")
    med = tab_df.median(axis=0, skipna=True)
    tab_df = tab_df.fillna(med).fillna(0.0)
    return np.ascontiguousarray(tab_df.to_numpy(dtype=np.float32))


def _prepare_target_matrix(y: pd.DataFrame, target_cols: list[str], n_samples: int) -> np.ndarray:
    yy = y.set_index("sample_id").reindex(np.arange(n_samples))
    yy_df = yy[target_cols].astype("float32")
    return np.ascontiguousarray(yy_df.to_numpy(dtype=np.float32))


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

    device = _device()
    use_cuda = device.type == "cuda"

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    batch_size = int(os.environ.get("NTLPOL_BATCH_SIZE", "4096" if use_cuda else str(cfg.get("batch_size", 256))))
    max_epochs = int(os.environ.get("NTLPOL_MAX_EPOCHS", "30" if use_cuda else str(cfg.get("max_epochs", 50))))
    patience = int(os.environ.get("NTLPOL_PATIENCE", "5" if use_cuda else str(cfg.get("early_stopping_patience", 8))))
    num_workers = int(os.environ.get("NTLPOL_NUM_WORKERS", "2" if use_cuda else "0"))

    train_cfg = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "learning_rate": float(os.environ.get("NTLPOL_LR", cfg.get("learning_rate", 1e-3))),
        "weight_decay": float(cfg.get("weight_decay", 1e-4)),
        "loss_weights": cfg.get(
            "loss_weights",
            {"percentile_reg": 0.5, "no_recovery_12m_cls": 0.35, "no_recovery_24m_cls": 0.35},
        ),
        "seq_hidden": int(cfg.get("seq_hidden", 64)),
        "tab_hidden": int(cfg.get("tab_hidden", 64)),
        "fusion_hidden": int(cfg.get("fusion_hidden", 128)),
        "dropout": float(cfg.get("dropout", 0.1)),
        "num_workers": num_workers,
        "amp": bool(int(os.environ.get("NTLPOL_AMP", "1" if use_cuda else "0"))),
        "log_every_batches": int(os.environ.get("NTLPOL_LOG_EVERY_BATCHES", "25")),
    }

    LOGGER.info("Loading processed arrays from %s", processed_dir)
    x_seq, x_tab, y, index = load_modeling_arrays(
        processed_dir=processed_dir,
        early_post_months=early_post_months,
    )
    splits = read_json(split_path)

    if index.empty or x_tab.empty or y.empty or not splits:
        pred = pd.DataFrame()
        metrics = pd.DataFrame()
        write_table(
            pred,
            output_dir / f"predictions/{encoder_type}_early{early_post_months}_predictions.parquet",
            index=False,
        )
        write_table(
            metrics,
            output_dir / f"metrics/{encoder_type}_early{early_post_months}_event_metrics.parquet",
            index=False,
        )
        LOGGER.warning("Empty data/splits. Wrote empty %s outputs.", encoder_type)
        return pred, metrics

    n_samples = int(x_seq.shape[0])

    raw_feature_cols = [c for c in x_tab.columns if c not in {"sample_id", "event_id", "grid_id"}]
    feature_cols = [c for c in raw_feature_cols if x_tab[c].notna().any()]
    dropped_cols = sorted(set(raw_feature_cols) - set(feature_cols))

    if dropped_cols:
        LOGGER.warning(
            "Dropped %s all-missing tabular feature columns: %s",
            len(dropped_cols),
            dropped_cols,
        )

    LOGGER.info(
        "Device=%s cuda=%s gpu=%s samples=%s x_seq=%s tab_features=%s target_cols=%s cfg=%s",
        device,
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        n_samples,
        tuple(x_seq.shape),
        len(feature_cols),
        TARGET_COLUMNS,
        train_cfg,
    )

    LOGGER.info("Preparing tabular and target matrices...")
    prep_t0 = time.time()
    x_tab_values = _prepare_tabular_matrix(x_tab, feature_cols, n_samples)
    y_values = _prepare_target_matrix(y, TARGET_COLUMNS, n_samples)
    LOGGER.info(
        "Prepared matrices in %.1fs: x_tab=%s y=%s",
        time.time() - prep_t0,
        x_tab_values.shape,
        y_values.shape,
    )

    all_preds = []
    rows = []
    y_meta = y[["sample_id", "event_id", "grid_id"] + TARGET_COLUMNS]

    for fold_idx, (fold, split) in enumerate(splits.items(), start=1):
        train_ids = [int(i) for i in split.get("train", [])]
        test_ids = [int(i) for i in split.get("test", [])]

        if not train_ids or not test_ids:
            LOGGER.warning("[%s] skipped empty split", fold)
            continue

        LOGGER.info(
            "===== Fold %s/%s: %s | train=%s test=%s =====",
            fold_idx,
            len(splits),
            fold,
            len(train_ids),
            len(test_ids),
        )

        pred = _train_one_fold(
            encoder_type=encoder_type,
            x_seq=x_seq,
            x_tab_values=x_tab_values,
            y_values=y_values,
            target_cols=TARGET_COLUMNS,
            train_ids=train_ids,
            test_ids=test_ids,
            feature_cols=feature_cols,
            cfg=train_cfg,
            model_path=output_dir / f"models/{encoder_type}/{encoder_type}_{fold}_early{early_post_months}.pt",
            fold=str(fold),
        )

        pred = pred.merge(y_meta, on="sample_id", how="left")
        pred["fold"] = fold
        all_preds.append(pred)

        metric_row = {"fold": fold, **evaluate_prediction_frame(pred)}
        rows.append(metric_row)
        LOGGER.info("[%s] metrics: %s", fold, metric_row)

    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    metrics_df = pd.DataFrame(rows)

    write_table(
        pred_df,
        output_dir / f"predictions/{encoder_type}_early{early_post_months}_predictions.parquet",
        index=False,
    )
    write_table(
        metrics_df,
        output_dir / f"metrics/{encoder_type}_early{early_post_months}_event_metrics.parquet",
        index=False,
    )

    LOGGER.info("Wrote %s predictions for %s samples", encoder_type, len(pred_df))
    return pred_df, metrics_df
