import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(description="Training Options for TSP with NI")

    # Overall settings
    parser.add_argument('--eval_only', action='store_true', help='used only if to evaluate a model')
    parser.add_argument('--graph_type', type=str, default='unif', help="The type (and size) of the problem graph. unif (uniform) or tsplib")
    parser.add_argument('--graph_size', type=int, default=20, help='Size of the problem graph (number of nodes)')

    # Training Params
    parser.add_argument('--il_size_range', type=int, nargs=2, default=[15, 25], help='size range for variable sized training graphs')
    parser.add_argument('--rl_size_range', type=int, nargs=2, default=[15, 25], help='size range for variable sized training graphs')
    parser.add_argument('--rl_size_sampling', type=str, default='uniform_stratified',
                        choices=['uniform_stratified', 'power_stratified', 'power_iid'],
                        help='How to sample RL graph sizes inside rl_size_range. '
                        'uniform_stratified preserves the current behavior; '
                             'power_stratified biases toward larger sizes while retaining stratification; '
                             'power_iid draws independent weighted samples, which is often a better fit when '
                             'sizes_per_update is small.')
    parser.add_argument('--rl_size_power', type=float, default=2.0,
                        help='Power exponent used by power-stratified or power-iid RL size sampling. '
                             'Larger values bias more strongly toward larger sizes.')
    parser.add_argument('--sizes_per_update', type=int, default=1, help='Number of different graph sizes per parameter update when training on variable graph sizes in train_v2 strategy')
    parser.add_argument('--warmup_mode', type=str, default='uniform', choices=['uniform', 'fixed', 'stratified'], help='Mode of the warmup t0.')

    parser.add_argument('--T_warm_mult', type=float, default=1.0, help='Max number of warm steps multiplier for training (T_warm_max = T_warm_mult * graph_size)')
    parser.add_argument('--batch_size', type=int, default=256, help='Fallback training batch size used by IL/RL when stage-specific batch sizes are not set')
    parser.add_argument('--il_batch_size', type=int, default=None, help='Batch size for imitation learning (defaults to --batch_size)')
    parser.add_argument('--rl_batch_size', type=int, default=None, help='Batch size for reinforcement/mixed training (defaults to --batch_size)')
    parser.add_argument('--epoch_size', type=int, default=128000, help='Number of instances per epoch during training')
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate ")
    parser.add_argument('--start_lr', type=float, default=None,
                        help='Force optimizer LR at the start of this run (overrides checkpoint LR if resuming).')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='Learning rate decay per epoch')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'bf16', 'fp16'],
                        help='Training precision mode. bf16/fp16 use CUDA autocast; fp16 also uses GradScaler.')
    parser.add_argument('--epoch_schedule', type=str, default='',
                        help="Custom epoch schedule, e.g. 'RL:10,IL:10,RL:20'. Overrides num_il_epochs/num_rl_epochs order.")

    # Imitation Learning
    parser.add_argument('--num_il_epochs', type=int, default=100, help='The number of IL epochs to train')
    parser.add_argument('--il_horizon', type=int, default=2, help='In IL the k optimal steps of the teacher.')
    parser.add_argument('--on_policy_IL', action='store_true', help='Use trained model to perform warmups.')

    # Reinforcement Learning
    parser.add_argument('--num_rl_epochs', type=int, default=200, help='The number of RL epochs to train')
    parser.add_argument('--rl_horizon', type=int, default=32, help='Number of steps to unroll for each RL training instance (trajectory length)')

    parser.add_argument('--ppo_epochs', type=int, default=1, help='Number of PPO epochs per update')
    parser.add_argument('--ppo_clip', type=float, default=0.2, help='PPO clip ratio')
    parser.add_argument('--grpo_group_size', type=int, default=20, help='Group size for GRPO (1 to disable)')
    parser.add_argument('--rl_group_rotations', type=int, default=5,
                        help='Number of evenly spaced coordinate rotations to instantiate within each RL group. '
                             'Must divide the effective GRPO group size. 1 disables rotation augmentation.')
    parser.add_argument('--rl_rotation_consistency_coef', type=float, default=0.03,
                        help='Coefficient for the RL rotation-consistency regularizer. '
                             '0 disables the auxiliary loss and preserves the current objective.')
    parser.add_argument('--init_cost_ref', type=str, default='warmup_best', choices=['warmup_best', 'warmup_last'], help='Reference cost for reward normalization')
    parser.add_argument('--rl_reward_norm', type=str, default='rel_init', choices=['rel_init', 'rel_current'],
                        help='RL reward normalization mode: use the initial warmup reference cost or accumulate per-improvement gains relative to the current incumbent cost.')
    parser.add_argument('--mc_candidate_size', type=int, default=8,
                        help='Maximum candidate actions per evaluated state for Monte Carlo branching; the sampled action is always included, <= 0 uses all valid candidates.')
    parser.add_argument('--mc_rollout_samples', type=int, default=4,
                        help='Number of Monte Carlo continuations per candidate action when estimating state-action values.')
    parser.add_argument('--mc_eps', type=float, default=1e-6,
                        help='Epsilon used in the incumbent-normalized Monte Carlo value target.')

    parser.add_argument('--update_old_model_every_batch', action='store_true', default=True, help='Update old policy model every batch in PPO')
    parser.add_argument('--update_old_model_freq', type=int, default=20, help='Frequency (in batches) to update old policy model in PPO if not updating every batch')

    # Reward normalization
    parser.add_argument('--no_adv_norm', action='store_true', help='Do not use advantage normalization')

    # NN Architecture
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3, help='Number of layers in the encoder/critic network')
    parser.add_argument('--n_heads_encoder', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu', 'relu_squared'], help='Activation function to use')
    parser.add_argument('--normalization', type=str, default='layer', choices=['batch', 'layer', 'instance', 'rms', 'none'], help='Normalization type to use')
    parser.add_argument('--zero_init_residuals', action='store_true', help='Use zero initialization for residual connections in attention layers')
    parser.add_argument('--logit_tanh_clip', type=float, default=10.0, help='Clip tanh on logits to this value to avoid large logit values')

    # Tour injection for TSP
    parser.add_argument('--not_use_tour_inject', action='store_true', help='Do not use tour injection for TSP')
    parser.add_argument('--tour_inject_alpha_init', type=float, default=0.1, help='Initial alpha value for tour injection')

    # Action history
    parser.add_argument('--tabu_mode', type=str, default='actions', choices=['actions', 'added_edges'],
                        help='Tabu memory mode for training and evaluation.')
    parser.add_argument('--tabu_action_tenure', type=int, default=8, help='Tabu tenure for action-pair memory.')
    parser.add_argument('--tabu_edge_tenure', type=int, default=2, help='Tabu tenure for edge-pair memory.')
    parser.add_argument('--not_use_action_hist_feats', action='store_true', help='Do not use action history features in the decoder')
    parser.add_argument('--use_added_edge_hist_feats', action='store_true', help='Use added-edge history features in the decoder.')
    parser.add_argument('--action_hist_len', type=int, default=16, help='Number of actions fed to the model')

    # Load models
    parser.add_argument('--load_path', default=None, help='Path to load model parameters and optimizer state from')
    parser.add_argument('--reset_optimizer_state_on_load', action='store_true',
                        help='When training from --load_path, clear loaded optimizer state (Adam moments) before epoch 0.')

    # Validation
    parser.add_argument('--val_graph_types', type=str, default=['unif', 'unif', 'unif', 'unif', 'tsplib', 'tsplib'], nargs='+', help="The type (and size) of the problem graph for validation. unif or tsplib")
    parser.add_argument('--val_graph_sizes', type=int, default=[20,     50,     100,    500,    20,       100], nargs='+', help='Size of the problem graph (number of nodes) for validation')
    parser.add_argument('--num_val_samples', type=int, default=[100,    100,     50,     10,    100,       50], nargs='+', help='Number of instances to use for validation')
    parser.add_argument('--eval_batch_size', type=int, default=100, help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--T_max_eval_mult', type=float, default=4.0, help='Max number of steps multiplier for evaluation (T_max = T_max_eval_mult * graph_size)')
    parser.add_argument(
        '--eval_time_budget_s_mult', '--T_max_eval_s_mult',
        dest='eval_time_budget_s_mult',
        type=float,
        default=0.0,
        help='Optional wall-clock evaluation budget multiplier '
             '(budget_s = eval_time_budget_s_mult * graph_size). 0 disables it.'
    )
    parser.add_argument('--eval_init_method', type=str, default='sequential', choices=['random', 'sequential', 'load'], help='Method to initialize the first node in evaluation')
    parser.add_argument('--eval_init_path', type=str, default=None,
                        help='Path to initial solutions for evaluation, --eval_init_method needs to be = "load". '
                             'May be a concrete file, a path template with {tag}/{graph_type}/{graph_size}, '
                             'or a tagged file like ...unif500... that will be auto-rewritten per validation tag if matching sibling files exist.')
    parser.add_argument('--eval_restarts', type=int, default=1, help="Parallel evaluation restarts")
    parser.add_argument('--save_full_trace', action='store_true', help='Save full trace in evaluation mode.')
    parser.add_argument('--save_restart_traces', action='store_true',
                        help='When saving multistart evaluation traces, also save per-restart trajectories in addition to the envelope trace.')
    parser.add_argument('--skip_train_validation', action='store_true',
                        help='Skip per-epoch validation during training to reduce runtime. Has no effect in --eval_only mode.')

    # Misc
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--save_dir', type=str, default='', help='Optional output directory override. In --eval_only mode, this controls where traces are saved.')
    parser.add_argument('--save', action='store_true', help='Save the model')
    parser.add_argument('--checkpoint_epochs', type=int, default=10, help='Save checkpoint every n epochs (default 10), 0 to save no checkpoints')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb to log experiment data per epoch')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to use')
    parser.add_argument('--verbose', action='store_true', help='Whether to print detailed information during training')
    parser.add_argument('--debug', action='store_true', help='Debug')
    parser.add_argument('--run_id', type=str, default='', help='Run ID')

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available()

    if opts.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {opts.batch_size}")
    if opts.il_horizon <= 0:
        raise ValueError(f"il_horizon must be > 0, got {opts.il_horizon}")
    if opts.rl_horizon <= 0:
        raise ValueError(f"rl_horizon must be > 0, got {opts.rl_horizon}")
    if opts.start_lr is not None and opts.start_lr <= 0:
        raise ValueError(f"start_lr must be > 0, got {opts.start_lr}")
    if opts.il_batch_size is not None and opts.il_batch_size <= 0:
        raise ValueError(f"il_batch_size must be > 0, got {opts.il_batch_size}")
    if opts.rl_batch_size is not None and opts.rl_batch_size <= 0:
        raise ValueError(f"rl_batch_size must be > 0, got {opts.rl_batch_size}")
    if opts.mc_rollout_samples <= 0:
        raise ValueError(f"mc_rollout_samples must be > 0, got {opts.mc_rollout_samples}")
    if opts.mc_eps <= 0:
        raise ValueError(f"mc_eps must be > 0, got {opts.mc_eps}")
    if opts.rl_group_rotations <= 0:
        raise ValueError(f"rl_group_rotations must be > 0, got {opts.rl_group_rotations}")
    if opts.rl_rotation_consistency_coef < 0:
        raise ValueError(
            f"rl_rotation_consistency_coef must be >= 0, got {opts.rl_rotation_consistency_coef}"
        )
    if opts.rl_size_power <= 0:
        raise ValueError(f"rl_size_power must be > 0, got {opts.rl_size_power}")
    if opts.eval_time_budget_s_mult < 0:
        raise ValueError(f"eval_time_budget_s_mult must be >= 0, got {opts.eval_time_budget_s_mult}")
    eff_grpo_group_size = max(1, int(opts.grpo_group_size))
    if opts.rl_group_rotations > eff_grpo_group_size:
        raise ValueError(
            f"rl_group_rotations ({opts.rl_group_rotations}) must be <= effective grpo_group_size ({eff_grpo_group_size})"
        )
    if eff_grpo_group_size % opts.rl_group_rotations != 0:
        raise ValueError(
            f"effective grpo_group_size ({eff_grpo_group_size}) must be divisible by "
            f"rl_group_rotations ({opts.rl_group_rotations})"
        )

    def _validate_range(name, value):
        lo, hi = map(int, value)
        if lo <= 0 or hi <= 0:
            raise ValueError(f"{name} values must be > 0, got {value}")
        if lo > hi:
            raise ValueError(f"{name} must satisfy min <= max, got {value}")

    _validate_range("rl_size_range", opts.rl_size_range)
    _validate_range("il_size_range", opts.il_size_range)

    if not (len(opts.val_graph_types) == len(opts.val_graph_sizes) == len(opts.num_val_samples)):
        raise ValueError(
            "Validation arguments must have matching lengths: "
            f"val_graph_types={len(opts.val_graph_types)}, "
            f"val_graph_sizes={len(opts.val_graph_sizes)}, "
            f"num_val_samples={len(opts.num_val_samples)}"
        )

    return opts
