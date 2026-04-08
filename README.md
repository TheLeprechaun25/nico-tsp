
# NICO: Neural Improvement for Combinatorial Optimization

This repository contains the implementation of **NICO** (*Neural Improvement for Combinatorial Optimization*) applied to the **Traveling Salesperson Problem (TSP)**.

NICO is a neural improvement framework that learns to iteratively refine candidate tours through local search. The codebase supports both training and evaluation across multiple TSP distributions and scales, including Euclidean and TSPLIB-style instances.

---

## Training

We provide example commands for training NICO at two target scales.

### NICO-20
Training on smaller instances with sizes up to 20:
```bash
python run.py \
  --use_wandb \
  --no_progress_bar \
  --sizes_per_update 1 \
  --batch_size 256 \
  --il_size_range 10 20 \
  --rl_size_range 10 20 \
  --update_old_model_every_batch \
  --update_old_model_freq 20 \
  --epoch_schedule IL:100,RL:200 \
  --save \
  --run_id NICO20
````

### NICO-100

Training on larger instances with sizes up to 100:

```bash
python run.py \
  --use_wandb \
  --no_progress_bar \
  --sizes_per_update 1 \
  --batch_size 256 \
  --il_size_range 20 50 \
  --rl_size_range 20 100 \
  --update_old_model_every_batch \
  --update_old_model_freq 20 \
  --epoch_schedule IL:100,RL:200 \
  --save \
  --run_id NICO100
```

---

## Evaluation

The following commands reproduce the evaluation settings used for NICO at different inference-time scaling budgets.

### NICO-100 (single run)

```bash
python run.py \
  --eval_only \
  --verbose \
  --load_path outputs/nico100/rl_epoch-200.pt \
  --val_graph_types unif unif unif unif tsplib \
  --val_graph_sizes 20 50 100 500 100 \
  --num_val_samples 100 100 100 100 100 \
  --T_max_eval_mult 10 \
  --save_full_trace \
  --save_restart_traces \
  --save_dir results/nico
```

### NICO-100 (8 restarts)

```bash
python run.py \
  --eval_only \
  --verbose \
  --load_path outputs/nico100/rl_epoch-200.pt \
  --eval_init_method random \
  --eval_restarts 8 \
  --val_graph_types unif unif unif unif tsplib \
  --val_graph_sizes 20 50 100 500 100 \
  --num_val_samples 100 100 100 100 100 \
  --T_max_eval_mult 10 \
  --save_full_trace \
  --save_restart_traces \
  --save_dir results/nico
```

### NICO-100 (32 restarts)

```bash
python run.py \
  --eval_only \
  --verbose \
  --load_path outputs/nico100/rl_epoch-200.pt \
  --eval_init_method random \
  --eval_restarts 32 \
  --val_graph_types unif unif unif unif tsplib \
  --val_graph_sizes 20 50 100 500 100 \
  --num_val_samples 100 100 100 100 100 \
  --T_max_eval_mult 10 \
  --save_full_trace \
  --save_restart_traces \
  --save_dir results/nico
```

---

## Evaluation from External Initial Solutions

NICO can also be used as a **post-hoc refinement module** by loading externally generated initial tours.

### NICO-100 with POMO initial solutions

```bash
python run.py \
  --eval_only \
  --verbose \
  --load_path outputs/nico100/rl_epoch-200.pt \
  --eval_init_method load \
  --eval_init_path 'results/pomo/anytime_results_policyN100_{tag}_seed55.pkl' \
  --val_graph_types unif unif unif tsplib \
  --val_graph_sizes 50 100 500 100 \
  --num_val_samples 100 100 100 100 \
  --T_max_eval_mult 10 \
  --save_full_trace \
  --save_dir results/nico/pomo_load
```

---

## Notes

* `--epoch_schedule IL:100,RL:200` denotes two-stage training with **100 imitation-learning epochs** followed by **200 reinforcement-learning epochs**.
* `--T_max_eval_mult 10` evaluates each instance with a search budget of **10n improvement steps**.
* `--eval_restarts k` runs NICO from `k` independent initial solutions and keeps the best result.
* `--eval_init_method load` allows NICO to refine tours produced by an external method.

---

## Output

Evaluation results, traces, and restart trajectories are saved under the directory specified by `--save_dir`.

---

## Citation

If you use this repository in academic work, please cite the corresponding paper.
