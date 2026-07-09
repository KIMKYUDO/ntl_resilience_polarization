from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero_loss_like(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), dtype=x.dtype, device=x.device)


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.float().reshape(-1)


def _masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float | torch.Tensor | None = None,
) -> torch.Tensor:
    logits_f = _flat(logits)
    target_f = _flat(target)

    mask = torch.isfinite(logits_f) & torch.isfinite(target_f)
    if int(mask.sum().item()) == 0:
        return _zero_loss_like(logits_f)

    logits_m = logits_f[mask]
    target_m = target_f[mask].clamp(0.0, 1.0)

    kwargs = {}
    if pos_weight is not None:
        kwargs["pos_weight"] = torch.as_tensor(
            float(pos_weight),
            dtype=logits_m.dtype,
            device=logits_m.device,
        )

    return F.binary_cross_entropy_with_logits(logits_m, target_m, **kwargs)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = _flat(pred)
    target_f = _flat(target)

    mask = torch.isfinite(pred_f) & torch.isfinite(target_f)
    if int(mask.sum().item()) == 0:
        return _zero_loss_like(pred_f)

    return F.mse_loss(pred_f[mask], target_f[mask])


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    alpha: float = 0.5,
    beta12: float = 0.35,
    beta24: float = 0.35,
    pos_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pos_weights = pos_weights or {}

    y_delayed = batch.get("y_delayed", batch.get("y_delayed_slowest_20pct"))
    y_percentile = batch.get("y_percentile", batch.get("recovery_delay_percentile"))
    y_no12 = batch.get("y_no12", batch.get("y_no_recovery_12m"))
    y_no24 = batch.get("y_no24", batch.get("y_no_recovery_24m"))

    if y_delayed is None:
        raise KeyError("Missing y_delayed target")
    if y_percentile is None:
        raise KeyError("Missing percentile target")
    if y_no12 is None:
        raise KeyError("Missing y_no12 target")
    if y_no24 is None:
        raise KeyError("Missing y_no24 target")

    delayed_loss = _masked_bce_with_logits(
        outputs["delayed_logit"],
        y_delayed,
        pos_weights.get("delayed"),
    )
    percentile_loss = _masked_mse(
        outputs["percentile"],
        y_percentile,
    )
    no12_loss = _masked_bce_with_logits(
        outputs["no_recovery_12m_logit"],
        y_no12,
        pos_weights.get("no12"),
    )
    no24_loss = _masked_bce_with_logits(
        outputs["no_recovery_24m_logit"],
        y_no24,
        pos_weights.get("no24"),
    )

    total = delayed_loss + alpha * percentile_loss + beta12 * no12_loss + beta24 * no24_loss

    parts = {
        "delayed": delayed_loss.detach(),
        "percentile": percentile_loss.detach(),
        "no12": no12_loss.detach(),
        "no24": no24_loss.detach(),
        "total": total.detach(),
    }
    return total, parts
