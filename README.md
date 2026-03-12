# NICO
Neural Improvement for Combinatorial Optimization (NICO). Applied to the Traveling Salesperson Problem (TSP)

## How to train

### NICO with n=20
python run.py --use_wandb --no_progress_bar --sizes_per_update 1 --batch_size 256 --il_size_range 10 20 --rl_size_range 10 20 --epoch_schedule IL:100,RL:400 --save --run_id NICO20

### NICO with n=100
python run.py --use_wandb --no_progress_bar --sizes_per_update 1 --batch_size 256 --il_size_range 20 50 --rl_size_range 20 100 --epoch_schedule IL:100,RL:400 --save --run_id NICO100


python run.py --tabu_mode added_edges --use_added_edge_hist_feats --use_wandb --no_progress_bar --sizes_per_update 1 --batch_size 256 --il_size_range 20 50 --rl_size_range 20 100 --epoch_schedule IL:100,RL:400 --save --run_id NICO100_edgetabu



## How to test
python run.py --eval_only --load_path /path/to/model.pt

python run.py --eval_only --verbose --T_max_eval_mult 10 --load_path outputs/nico100/rl_epoch-110.pt


## TO-DOs
- [ ] Check tabu actions are correctly masked.
- [ ] Scale problem. Sample instances with a freq inverse to the obtained reward scale
- [ ] Credit Assignment problem. use montecarlo rollouts to estimate the value of each action. 
       Avoid repeating states and doing wastefull work. keep a number of possible samples from each state and sample that many actions.
- [ ] Training efficiency. Collect different states of the same instance with the reference policy.




