# Combinatorial restless bandits

This code implements and evaluates algorithms for the paper [**Reinforcement learning with combinatorial actions for coupled restless bandits**](https://arxiv.org/abs/2503.01919) by Lily Xu, Bryan Wilder, Elias B. Khalil, and Milind Tambe which appeared at the International Conference on Learning Representations (ICLR 2025). In this paper we consider reinforcement learning (RL) with combinatorial actions, specifically for restless bandits.

Typical restless bandits assume a simple budget constraint, which allow planning to be decoupled across each arm to learn an optimal policy for each arm separately, then using a simple threshold-based heuristic to select the optimal action. 

Here, we introduce **coRMAB**, a broader class of problems with _combinatorial actions_ that cannot be decoupled across the arms of the restless bandit, requiring direct solving over the joint, exponentially large action space. We propose **SEQUOIA**, an algorithm which directly solves for long-term reward by embedding a Q-network into a mixed-integer linear program (MILP) to select a combinatorial action in each timestep. 

In this codebase, we implement and evaluate SEQUOIA. We include a number of simulation environments introduced in the paper, including:

1. path-constrained,
2. schedule-constrained,
3. capacity-constrained, and
4. multiple interventions.

This library also implements and compares against a variety of baselines:  null (no action), random, sampling, myopic, iterative myopic, iterative DQN.

Details about all the environments and their implementation are in the paper appendix.


## Code execution

Running `driver.py` runs all four domains in sequence.

The following options are available:

*  `-H` horizon, default $H=20$
*  `-J` number of arms, default $J=20$
*  `-B` budget, default $B=8$
*  `-N` number of actions (used for multiple interventions setting), default $N=12$
*  `-D` type of RMAB setting {`sigmoid`, `scheduling`, `constrained`, `routing`}, default `constrained`
*  `-p` prefix for output files, default empty `''`
*  `-s` random seed, default $s=0$
*  `-K` number of samples (for sampling baseline), default $K=100$
*  `-V` number of episodes, for evaluation, default $V=50$
*  `-F` avoid training the model (for fast prototyping only), default `False`


## Domains

The four domains of coRMAB implemented here are:


###  **Path-constrained**  (`RoutingRMAB` in `routing.py`)  

Each arm lies within a network, and we must find a valid route.

Network from London underground map: https://commons.wikimedia.org/wiki/London_Underground_geographic_maps

```
python driver.py -s 0 -H 20 -J 20 -B 5 -p run_note --rmab_type routing -V 1 -K 10 -F

python driver.py -s 0 -H 20 -J 40 -B 10 -p run_note --rmab_type routing -V 50 -K 100
```


###  **Schedule-constrained** (`SchedulingRMAB` in `scheduling.py`)

Each arm and worker are available for a limited number of scheduling slots out of $K$ total slots.

```
python driver.py -s 0 -H 5 -J 5 -B 4 -p run_note --rmab_type scheduling -V 1 -K 10 -F

python driver.py -s 0 -H 20 -J 20 -B 5 -p run_note --rmab_type scheduling -V 50 -K 100

python driver.py -s 0 -H 20 -J 100 -B 20 -p run_note --rmab_type scheduling -V 50 -K 100
```


### **Capacity-constrained** (`ConstrainedRMAB` in `constrained.py`)

Each worker is limited by a capacity constriant of how many arms they can pull; each arm has a differing cost.

```
python driver.py -s 0 -H 5 -J 5 -B 4 -p run_note --rmab_type constrained -V 1 -K 10 -F

python driver.py -s 0 -H 20 -J 20 -B 5 -p run_note --rmab_type constrained -V 50 -K 100

python driver.py -s 0 -H 20 -J 100 -B 20 -p run_note --rmab_type constrained -V 50 -K 100
```



### **Multiple interventions**  (`MultiActionRMAB` in `multi_action.py`)

The multiple-invention RMAB setting is implemented in `multi_action.py` as `MultiActionRMAB` with `link_type='sigmoid'`.

The effects of these multiple cumulative actions are modeled as a sigmoid link function. The piecewise-linear version of the sigmoid function used is specified in `sigmoid_alpha4_beta2_x3_breaks10.pickle`.

Example execution:
```
python driver.py -s 0 -H 5 -J 5 -N 8 -B 4 -p run_note -V 1 -K 10 --rmab_type sigmoid -F

python driver.py -s 0 -H 20 -J 20 -N 10 -B 5 -p run_note -V 50 -K 100 --rmab_type sigmoid
```


## Baselines

We compare against the following baselines:

1. **Null** (no action): `baseline_null_action()` in `baseline.py`
1. **Random**: `baseline_random()` in `baseline.py`
1. **Sampling**: `baseline_sample()` in `baseline.py`
1. **Myopic**: `baseline_myopic()` in `baseline.py`
1. **Iterative myopic**: `baseline_greedy_iterative_myopic()` in `baseline.py`
1. **Iterative DQN**: `baseline_iterative_dqn()` in `baseline_iterative_dqn.py`


## Other key files

* `run.sh` executes repeated versions of different problem instances
* `MultiStateRMAB` is the "base" restless bandit used in all experiments here
* `environment/bipartite.py` creates and visualizes bipartite graphs
* `environment/rmab_instances.py` generates the various RMAB problem instances
* `./approximator/*_approximator.py`  implements the MILP solvers for each problem setting
* `./model2mip/*` files in this folder implement the process of embedding a trained neural network into a MILP, from [Neur2SP](https://github.com/khalil-research/Neur2SP)

  
  

## Dependencies

This code was developed using Python version `3.10.14`. We require the following packages:

- numpy==1.24.3
- pandas==2.0.3
- scipy==1.9.3
- matplotlib==3.7.1
- pytorch==2.0.0
- networkx==3.3
- gurobi==10.0.1


## References

We build off Neur2SP from the work of [Dumouchelle et al.](https://arxiv.org/pdf/2205.12006), whose implementation for embedding a trained neural network into a MILP are in the `./model2mip/` folder. Their code is available at https://github.com/khalil-research/Neur2SP



## Citing this paper

```
@inproceedings{xu2025reinforcement,
  title={Reinforcement learning with combinatorial actions for coupled restless bandits},
  author={Xu, Lily and Wilder, Bryan and Khalil, Elias B. and Tambe, Milind},
  booktitle={Proceedings of the 13th International Conference on Learning Representations},
  year={2025},
}
```
