"""Muon + AdamW in one torch Optimizer (train.optimizer = "muon_adamw").

Same split and update rule as the NeuroQAP training code: every 2-D weight matrix is updated with
Muon (momentum 0.95, Nesterov, 5 Newton-Schulz steps, ``adjust_lr_fn="match_rms_adamw"`` so the update
RMS is ~0.2 and both parts share one learning rate); all other parameters (biases, norms, scalars)
with AdamW. The Muon step is a line-for-line port of torch 2.13's ``torch/optim/_muon.py``
(single-tensor path). One Optimizer subclass, so learning-rate schedulers, ``state.clear()`` and
``state_dict()`` work as for AdamW.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


def _zeropower_via_newtonschulz(grad, ns_coefficients, ns_steps, eps):
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))  # spectral norm <= 1
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


def _adjust_lr(lr, adjust_lr_fn, param_shape):
    A, B = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(A, B))
    else:
        ratio = 1.0
    return lr * ratio


class MuonAdamW(Optimizer):
    def __init__(self, params, lr: float, weight_decay: float = 0.01, momentum: float = 0.95,
                 nesterov: bool = True, ns_coefficients=(3.4445, -4.775, 2.0315), ns_steps: int = 5,
                 eps: float = 1e-7, betas=(0.9, 0.999), adam_eps: float = 1e-8,
                 adjust_lr_fn: str = "match_rms_adamw"):
        params = [p for p in params if p.requires_grad]
        groups = [{"params": [p for p in params if p.ndim == 2], "use_muon": True},
                  {"params": [p for p in params if p.ndim != 2], "use_muon": False}]
        groups = [g for g in groups if g["params"]]
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                        ns_coefficients=ns_coefficients, ns_steps=ns_steps, eps=eps, betas=betas,
                        adam_eps=adam_eps, adjust_lr_fn=adjust_lr_fn)
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if group["use_muon"]:
                    mom = group["momentum"]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                    buf = state["momentum_buffer"]
                    buf.lerp_(grad, 1 - mom)
                    update = grad.lerp(buf, mom) if group["nesterov"] else buf
                    update = _zeropower_via_newtonschulz(update, group["ns_coefficients"], group["ns_steps"],
                                                         group["eps"])
                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-_adjust_lr(lr, group["adjust_lr_fn"], p.shape))
                else:   # AdamW (decoupled weight decay), as torch.optim.AdamW
                    b1, b2 = group["betas"]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    p.mul_(1 - lr * wd)
                    state["exp_avg"].lerp_(grad, 1 - b1)
                    state["exp_avg_sq"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                    denom = (state["exp_avg_sq"].sqrt() / math.sqrt(1 - b2 ** t)).add_(group["adam_eps"])
                    p.addcdiv_(state["exp_avg"], denom, value=-lr / (1 - b1 ** t))
        return loss
