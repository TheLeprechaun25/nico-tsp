"""Train NICO.

    python scripts/train.py --config tsp/configs/nico_default.yaml
    python scripts/train.py --config cvrp/configs/nico_cvrp_default.yaml train.seed=43 run_name=cvrp_s43
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nico.config import Config  # noqa: E402
from nico.trainer import train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default=None, help="log epoch metrics to this W&B project")
    parser.add_argument("--resume", action="store_true", help="continue from <output_dir>/<run_name>/last.pt")
    parser.add_argument("overrides", nargs="*", help="key.sub=value overrides of the config")
    args = parser.parse_args()

    cfg = Config.load(args.config, args.overrides)
    # TF32 matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    log = None
    if args.wandb_project:
        import wandb

        # W&B refuses ids of deleted runs: WANDB_RUN_ID overrides the default id (= run_name)
        run_id = os.environ.get("WANDB_RUN_ID", cfg.run_name)
        wandb.init(project=args.wandb_project, name=cfg.run_name, id=run_id, resume="allow",
                   config=cfg.to_dict())
        log = wandb.log
    _, out_dir = train(cfg, torch.device(args.device), log=log, resume=args.resume)
    print(f"Done. Checkpoints and metrics in {out_dir}")


if __name__ == "__main__":
    main()
