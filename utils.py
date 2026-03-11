import json
import math
import os
import re
from contextlib import nullcontext
import torch
from argparse import Namespace


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def get_inner_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def move_to(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to(v, device) if v is not None else None for k, v in x.items()}
    return x


def autocast_context(opts):
    device = getattr(opts, "device", torch.device("cpu"))
    precision = str(getattr(opts, "precision", "fp32")).lower()
    if not isinstance(device, torch.device):
        device = torch.device(str(device))
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()

    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(opts):
    device = getattr(opts, "device", torch.device("cpu"))
    precision = str(getattr(opts, "precision", "fp32")).lower()
    enabled = isinstance(device, torch.device) and device.type == "cuda" and precision == "fp16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param param_groups:
    :param max_norm:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def reset_optimizer_state(optimizer):
    """
    Clear per-parameter optimizer statistics (e.g., momentum / Adam moments).
    Keeps param groups and current learning rates unchanged.
    """
    if optimizer is None:
        return 0
    n_state_entries = len(optimizer.state)
    optimizer.state.clear()
    return n_state_entries


def ns_to_dict(x):
    if isinstance(x, Namespace):
        return {k: ns_to_dict(v) for k, v in vars(x).items()}
    if isinstance(x, dict):
        return {k: ns_to_dict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(ns_to_dict(v) for v in x)
    return x


_MODEL_ARG_KEYS_CACHE = None


def _infer_model_arg_keys_from_options(options_path: str):
    keys = set()
    if not options_path or not os.path.exists(options_path):
        return tuple()

    model_sections = (
        "# NN Architecture",
        "# MoE",
        "# Tour injection",
        "# Action history",
    )

    in_section = False
    with open(options_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                if any(marker in stripped for marker in model_sections):
                    in_section = True
                    continue
                if in_section:
                    # Exit model-related sections on the next non-model header.
                    if not any(marker in stripped for marker in model_sections):
                        in_section = False
            if not in_section:
                continue
            m = re.search(r"add_argument\(\s*['\"]--([A-Za-z0-9_-]+)", line)
            if m:
                keys.add(m.group(1).replace("-", "_"))
    return tuple(sorted(keys))


def get_model_arg_keys():
    global _MODEL_ARG_KEYS_CACHE
    if _MODEL_ARG_KEYS_CACHE is None:
        options_path = os.path.join(os.path.dirname(__file__), "options.py")
        _MODEL_ARG_KEYS_CACHE = _infer_model_arg_keys_from_options(options_path)
    return _MODEL_ARG_KEYS_CACHE


def build_model_args(opts, keys=None):
    if keys is None:
        keys = get_model_arg_keys()
    out = {}
    for k in keys:
        if hasattr(opts, k):
            out[k] = getattr(opts, k)
    return out


def maybe_apply_checkpoint_args(opts, load_path=None, load_data=None, keys=None, verbose=False):
    if keys is None:
        keys = get_model_arg_keys()

    saved_args = {}
    if isinstance(load_data, dict):
        model_args = load_data.get("model_args")
        if isinstance(model_args, dict):
            saved_args = model_args

    if not saved_args and load_path:
        load_path = os.path.expanduser(str(load_path))
        ckpt_dir = load_path if os.path.isdir(load_path) else os.path.dirname(load_path)
        args_path = os.path.join(ckpt_dir, "args.json")
        if os.path.exists(args_path):
            with open(args_path, "r") as f:
                saved_args = json.load(f)

    if not saved_args:
        return {}

    applied = {}
    for k in keys:
        if k in saved_args and hasattr(opts, k):
            cur = getattr(opts, k)
            new = saved_args[k]
            if cur != new:
                setattr(opts, k, new)
                applied[k] = new

    if verbose and applied:
        src = "checkpoint model_args" if isinstance(load_data, dict) and "model_args" in load_data else "args.json"
        keys_str = ", ".join(sorted(applied.keys()))
        print(f"  [*] Applied {src} for: {keys_str}")

    return applied


def parse_epoch_schedule(schedule: str):
    if schedule is None:
        return []
    schedule = str(schedule).strip()
    if not schedule:
        return []

    parts = [p.strip() for p in schedule.split(",") if p.strip()]
    if not parts:
        return []

    out = []
    for part in parts:
        m = re.match(r"^(il|rl)\s*[:x*]?\s*(\d+)$", part, re.IGNORECASE)
        if not m:
            raise ValueError(
                f"Invalid epoch_schedule entry '{part}'. Use e.g. 'IL:10,RL:20'."
            )
        stage = m.group(1).lower()

        count = int(m.group(2))
        if count <= 0:
            raise ValueError(f"Epoch schedule counts must be positive, got '{part}'.")
        out.extend([stage] * count)
    return out


def resolve_epoch_schedule(opts):
    schedule = parse_epoch_schedule(getattr(opts, "epoch_schedule", None))
    if schedule:
        opts.epoch_schedule_is_custom = True
        opts.num_il_epochs = int(sum(1 for s in schedule if s == "il"))
        opts.num_rl_epochs = int(sum(1 for s in schedule if s == "rl"))
        opts.total_epochs = int(len(schedule))
        return schedule

    opts.epoch_schedule_is_custom = False
    num_il = int(getattr(opts, "num_il_epochs", 0))
    num_rl = int(getattr(opts, "num_rl_epochs", 0))
    opts.total_epochs = int(num_il + num_rl)
    return ["il"] * num_il + ["rl"] * num_rl
