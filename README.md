# NICO
Neural Improvement for Combinatorial Optimization (NICO). Applied to the Traveling Salesperson Problem (TSP)

## How to train

### NICO with n=20
python run.py --use_wandb --no_progress_bar --sizes_per_update 1 --batch_size 256 --il_size_range 10 20 --rl_size_range 10 20 --epoch_schedule IL:100,RL:400 --save --run_id NICO20

### NICO with n=100
python run.py --use_wandb --no_progress_bar --sizes_per_update 1 --batch_size 256 --il_size_range 20 50 --rl_size_range 20 100 --epoch_schedule IL:100,RL:400 --save --run_id NICO100


## How to test

### NICO100 x1

python run.py \
  --eval_only \
  --verbose \
  --load_path outputs/508-215-nico100/rl_epoch-160.pt \
  --val_graph_types unif unif unif unif tsplib \
  --val_graph_sizes 20 50 100 500 100 \
  --num_val_samples 100 100 100 100 100 \
  --T_max_eval_mult 10 \
  --save_full_trace \
  --save_restart_traces \
  --save_dir results/nico

### NICO100 x8

python run.py \
  --eval_only \
  --verbose \
  --load_path outputs/508-215-nico100/rl_epoch-160.pt \
  --eval_init_method random \
  --eval_restarts 8 \
  --val_graph_types unif unif unif unif tsplib \
  --val_graph_sizes 20 50 100 500 100 \
  --num_val_samples 100 100 100 100 100 \
  --T_max_eval_mult 10 \
  --save_full_trace \
  --save_restart_traces \
  --save_dir results/nico

### NICO100 x32

python run.py \
  --eval_only \
  --verbose \
  --load_path outputs/508-215-nico100/rl_epoch-160.pt \
  --eval_init_method random \
  --eval_restarts 32 \
  --val_graph_types unif unif unif unif tsplib \
  --val_graph_sizes 20 50 100 500 100 \
  --num_val_samples 100 100 100 100 100 \
  --T_max_eval_mult 10 \
  --save_full_trace \
  --save_restart_traces \
  --save_dir results/nico



### NICO100 with loaded initial solutions - POMO

python run.py \
  --eval_only \
  --verbose \
  --load_path outputs/508-215-nico100/rl_epoch-160.pt \
  --eval_init_method load \
  --eval_init_path 'results/pomo/anytime_results_policyN100_{tag}_seed55.pkl' \
  --val_graph_types unif unif unif tsplib
  --val_graph_sizes 50 100 500 100
  --num_val_samples 100 100 100 100
  --T_max_eval_mult 10 \
  --save_full_trace \
  --save_dir results/nico/pomo_load

