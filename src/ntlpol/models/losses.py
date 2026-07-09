from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, pos_weight=None) -> torch.Tensor:
    mask = torch.isfinite(targets)
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logits[mask], targets[mask], pos_weight=pos_weight
    )


def masked_huber(pred: torch.Tensor, targets: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    mask = torch.isfinite(targets)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return F.huber_loss(pred[mask], targets[mask], delta=delta)


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    alpha: float = 0.5,
    beta12: float = 0.35,
    beta24: float = 0.35,
    pos_weights: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    pos_weights = pos_weights or {}
    loss_delayed = masked_bce_with_logits(
        outputs["delayed_logit"], batch["y_delayed"], pos_weights.get("delayed")
    )
    loss_pct = masked_huber(outputs["percentile"], batch["y_percentile"])
    loss_no12 = masked_bce_with_logits(
        outputs["no_recovery_12m_logit"], batch["y_no12"], pos_weights.get("no12")
    )
    loss_no24 = masked_bce_with_logits(
        outputs["no_recovery_24m_logit"], batch["y_no24"], pos_weights.get("no24")
    )
    total = loss_delayed + alpha * loss_pct + beta12 * loss_no12 + beta24 * loss_no24
    return total, {
        "loss": float(total.detach().cpu()),
        "loss_delayed": float(loss_delayed.detach().cpu()),
        "loss_percentile": float(loss_pct.detach().cpu()),
        "loss_no12": float(loss_no12.detach().cpu()),
        "loss_no24": float(loss_no24.detach().cpu()),
    }
