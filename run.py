import os
import json
import datetime
import random
import torch
import torch.optim as optim

from options import get_options
from problem_tsp import TSP
from model import EdgeAttentionModel
from train_imitation_learning import train_il_epoch
from train_reinforcement_learning import train_rl_epoch
from utils import (
    torch_load_cpu,
    get_inner_model,
    ns_to_dict,
    resolve_epoch_schedule,
    maybe_apply_checkpoint_args,
    reset_optimizer_state,
    make_grad_scaler,
)
from validation import validate

wandb = None
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def _apply_debug_overrides(opts):
    if getattr(opts, "debug", False):
        opts.val_graph_types = ['unif']
        opts.val_graph_sizes = [20]
        opts.num_val_samples = [10]
        opts.verbose = True
        opts.save_dir = None
        opts.batch_size = 16
        opts.epoch_size = 64
        opts.num_il_epochs = 2
        opts.num_rl_epochs = 2

    else:
        requested_save_dir = str(getattr(opts, "save_dir", "") or "").strip()
        if opts.save:
            if requested_save_dir:
                opts.save_dir = requested_save_dir
            else:
                cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                run_id = opts.run_id if opts.run_id else "run"
                opts.save_dir = f"outputs/{run_id}_{cur_time}"
        else:
            opts.save_dir = requested_save_dir if requested_save_dir else None


def run(opts):
    opts.debug = True
    _apply_debug_overrides(opts)
    schedule = resolve_epoch_schedule(opts)

    # Set the random seed
    seed = int(opts.seed)
    opts.process_seed = seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize the problem class
    problem = TSP(opts=opts)

    if opts.save_dir is not None and opts.save_dir != "":
        if not os.path.exists(opts.save_dir):
            os.makedirs(opts.save_dir)

        # Save arguments so exact configuration can always be found
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    if opts.use_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project="NI_TSP",
                   name=f"{opts.run_id}_{opts.seed}",
                   config=ns_to_dict(opts),
                   dir=opts.save_dir if opts.save_dir != "" else None)
        # Two x-axes
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val_*", step_metric="epoch")

    # Set the device
    if opts.use_cuda:
        opts.device = torch.device("cuda")
    else:
        opts.device = torch.device("cpu")

    # Load model data from load_path
    load_data = {}
    if opts.load_path is not None:
        if getattr(opts, "is_main_process", True):
            print('  [*] Loading checkpoint from {}'.format(opts.load_path))
        load_data = torch_load_cpu(opts.load_path)

    # Apply model hyperparameters saved with the checkpoint if available
    maybe_apply_checkpoint_args(opts, load_path=opts.load_path, load_data=load_data, verbose=opts.verbose)

    # Initialize model
    model = EdgeAttentionModel(
        opts=opts,
        problem=problem,
        embedding_dim=opts.embedding_dim,
        hidden_dim=opts.hidden_dim,
        n_heads=opts.n_heads_encoder,
        n_layers=opts.n_encode_layers,
        normalization=opts.normalization,
        device=opts.device
    ).to(opts.device)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    val_datasets = []
    should_load_val = opts.eval_only or (not getattr(opts, "skip_train_validation", False))
    if should_load_val:
        for graph_type, graph_size, n_samples in zip(opts.val_graph_types, opts.val_graph_sizes, opts.num_val_samples):
            val_datasets.append(problem.make_dataset(opts=opts, graph_type=graph_type, graph_size=graph_size, num_samples=n_samples, test=True))

    # Do validation only
    if opts.eval_only:
        run_id = opts.load_path.split('/')[1].split('_')[0] if opts.load_path is not None else "debug"
        if opts.save_dir is None or opts.save_dir == "":
            opts.save_dir = f"results/nico/nico_{run_id}"

        validate(problem, model, val_datasets, opts, save_full_trace=opts.save_full_trace)

    # Do training
    else:
        opts._grad_scaler = make_grad_scaler(opts)

        # Initialize optimizer
        param_groups = [{'params': model.parameters(), 'lr': opts.lr_model}]
        optimizer = optim.AdamW(param_groups)

        # Load optimizer state
        if 'optimizer' in load_data:
            optimizer.load_state_dict(load_data['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(opts.device)

            if getattr(opts, "reset_optimizer_state_on_load", False):
                cleared = reset_optimizer_state(optimizer)
                if opts.verbose:
                    print(
                        f"[opt] Reset optimizer state after checkpoint load "
                        f"(cleared {cleared} entries)",
                        flush=True,
                    )

        if getattr(opts, "start_lr", None) is not None:
            forced_lr = float(opts.start_lr)
            for group in optimizer.param_groups:
                group["lr"] = forced_lr
                if "initial_lr" in group:
                    group["initial_lr"] = forced_lr
            if opts.verbose:
                print(f"[opt] Overriding optimizer start lr to {forced_lr:.3e}", flush=True)

        # Initialize learning rate scheduler, decay by lr_decay once per epoch
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opts.lr_decay)

        # Start the actual training loop
        prev_stage = None
        for epoch, stage in enumerate(schedule):
            if prev_stage is not None and stage != prev_stage:
                cleared = reset_optimizer_state(optimizer)
                if opts.verbose:
                    print(
                        f"[opt] Reset optimizer state on stage change "
                        f"{prev_stage.upper()} -> {stage.upper()} (cleared {cleared} entries)",
                        flush=True,
                    )

            if stage == "il":
                # Stage 1: Imitation Learning
                epoch_results = train_il_epoch(
                    problem, model, optimizer, epoch, val_datasets, opts
                )
            else:
                # Stage 3: Reinforcement Learning
                epoch_results = train_rl_epoch(
                    problem, model, optimizer, epoch, val_datasets, opts
                )
            # Step the learning rate scheduler
            lr_scheduler.step()
            prev_stage = stage

            # Store results
            if opts.use_wandb and getattr(opts, "is_main_process", True):
                epoch_results["stage"] = stage
                wandb.log(epoch_results)


if __name__ == "__main__":
    run(get_options())
