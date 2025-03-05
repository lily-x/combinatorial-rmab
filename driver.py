""" to run:
> conda activate causal
> python driver.py
> python driver.py -s 0 -H 20 -J $N_ARMS -B $BUDGET -p $PREFIX -D $RMAB_TYPE

or 
> ./run.sh
"""

import os, sys
import argparse
import datetime

from collections import OrderedDict
import random
import numpy as np
import pandas as pd
from scipy import stats
import torch
import matplotlib.pyplot as plt

from environment.rmab_instances import get_rmab_sigmoid, get_scheduling, get_constrained, get_routing
from environment.graph_utils import visualize_graph

from environment.base_envs import StandardRMAB, TorchStandardRMAB
from environment.multi_action import MultiActionRMAB
from environment.constrained import ConstrainedRMAB, TorchConstrainedRMAB
from environment.scheduling import SchedulingRMAB, TorchSchedulingRMAB
from environment.routing import RoutingRMAB

from algos.dqn_estimator import DQNSolver
from algos.baselines import baseline_null_action, baseline_random, baseline_myopic, baseline_sample, baseline_greedy_iterative_myopic
from algos.baselines import baseline_optimal
from algos.baseline_iterative_dqn import baseline_iterative_dqn
from algos.evaluate_MIP import MIP_results

from model2mip.net2mip import Net2MIPPerScenario


out_dir = './plots'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', help='random seed', type=int, default=0)
    parser.add_argument('--horizon', '-H', help='time horizon', type=int, default=20)
    parser.add_argument('--rmab_type', '-D', help='rmab type {sigmoid, multistate, scheduling, constrained, standard, routing}', type=str, default='multistate')  # also linear, submodular, sigmoid
    parser.add_argument('--n_actions', '-N', help='number of actions', type=int, default=3)
    parser.add_argument('--n_arms', '-J', help='number of arms', type=int, default=5)
    parser.add_argument('--budget', '-B', help='budget', type=int, default=2)
    parser.add_argument('--n_samples', '-K', help='number of samples (for sampling baseline)', type=int, default=100)
    parser.add_argument('--n_episodes_eval', '-V', help='number of episodes to run for evaluation', type=int, default=50)
    parser.add_argument('--prefix', '-p', help='prefix for file writing', type=str, default='')
    parser.add_argument('--fast_prototyping', '-F', help='if True, then skip many steps to just run code (default False)', action='store_true')
    parser.add_argument('--no_dqn', '-Q', help='if True, then skip the DQN steps (default False)', action='store_true')
    # parser.add_argument('--num_repeats', '-R', help='number of repeats', type=int, default=30)
    # # parser.add_argument('--num_processes', '-P', help='number of processes', type=int, default=4)
    # parser.add_argument('--verbose', '-V', help='if True, then verbose output (default False)', action='store_true')
    # parser.add_argument('--local', '-L', help='if True, running locally (default False)', action='store_true')
    # # parser.add_argument('--synthetic', '-S', help='if True, then use synthetic data; else real-world (default)', action='store_true')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    horizon = args.horizon
    rmab_type = args.rmab_type
    n_actions = args.n_actions
    n_arms = args.n_arms
    budget = args.budget
    n_samples = args.n_samples
    n_episodes_eval = args.n_episodes_eval
    prefix = args.prefix
    fast_prototyping = args.fast_prototyping

    run_multiple = False #True
    
    # other baselines (turn off for speedy prototyping)
    no_dqn = args.no_dqn
    use_sample = True

    if rmab_type in ['linear', 'submodular', 'sigmoid']:
        assert n_actions is not None

    print('--------------------------------------------------------')
    print('get RMAB graph')
    print('--------------------------------------------------------')
    
    edge_prob = 0.1

    if rmab_type == 'sigmoid':
        rmab, t_rmab = get_rmab_sigmoid(n_arms, n_actions, budget, horizon)

        print('graph stats')
        print(f'  dim: {rmab.action_dim} actions, {rmab.n_arms} arms')
        print(f'  action degree: {(rmab.weights_p > 0).sum(axis=0)}')
        print(f'  arm degree: {(rmab.weights_p > 0).sum(axis=1)}')

    # constrained ------------------------------------
    elif rmab_type == 'constrained':
        inner_type = 'multistate'
        rmab, t_rmab = get_constrained(horizon, n_arms, budget, rmab_type = inner_type)

    # scheduling ------------------------------------
    elif rmab_type == 'scheduling':
        inner_type = 'multistate'
        rmab, t_rmab = get_scheduling(horizon, n_arms, budget, rmab_type = inner_type)

    # routing ------------------------------------
    elif rmab_type == 'routing':
        inner_type = 'multistate'
        rmab, t_rmab = get_routing(horizon, n_arms, budget, rmab_type = inner_type)

    else:
        raise NotImplementedError
    
    
    if isinstance(rmab, MultiActionRMAB):
        rmab_type = f'MultiActionRMAB-{rmab.link_type}'
        # if rmab.n_arms < 30:
        #     visualize_graph(rmab, out_dir)
        #     # plt.show()
        #     plt.close()

    init_states = [np.random.binomial(1, 0.2, rmab.n_arms) for _ in range(n_episodes_eval)]
    print(f'running {rmab} with prefix {prefix}')


    start_time = datetime.datetime.now()
    if not no_dqn:
        print('--------------------------------------------------------')
        print('train DQN solver')
        print('--------------------------------------------------------')
        dqn_solver = DQNSolver(t_rmab)
        
        dqn_net, myopic_net, midway_net = dqn_solver.train(fast_prototyping=fast_prototyping)
    dqn_end_time = datetime.datetime.now()

    if run_multiple:
        con_rmab = ConstrainedRMAB(rmab)
        con_t_rmab = TorchConstrainedRMAB(t_rmab, con_rmab.arm_costs, con_rmab.worker_capacity)
        n_timeslots = 5
        sch_rmab = SchedulingRMAB(rmab, n_timeslots=n_timeslots)
        sch_t_rmab = TorchSchedulingRMAB(t_rmab, n_timeslots=sch_rmab.n_timeslots, arm_M=sch_rmab.arm_M, worker_M=sch_rmab.worker_M, arm_avail=sch_rmab.arm_avail, worker_avail=sch_rmab.worker_avail)
        path_rmab = RoutingRMAB(rmab)
        path_t_rmab = RoutingRMAB(t_rmab, G=path_rmab.G, G_pos=path_rmab.G_pos, edges=path_rmab.edges, edge_list=path_rmab.edge_list, source=path_rmab.source, valid_cycles=path_rmab.valid_cycles)
        use_rmab = [(rmab, t_rmab), (con_rmab, con_t_rmab), (sch_rmab, sch_t_rmab), (path_rmab, path_t_rmab)]
    else:
        use_rmab = [(rmab, t_rmab)]


    for (rmab, t_rmab) in use_rmab:
        print('--------------------------------------------------------')
        print('run baselines')
        print('--------------------------------------------------------')

        algo_rewards = OrderedDict()

        algo_rewards['null'] = baseline_null_action(rmab, init_states)
        algo_rewards['random'] = baseline_random(rmab, init_states)
        if use_sample: algo_rewards[f'sampling (k={n_samples})'] = baseline_sample(rmab, init_states, n_samples=n_samples)
        algo_rewards['iterative myopic'] = baseline_greedy_iterative_myopic(rmab, init_states)

        print('--------------------------------------------------------')
        print('myopic')
        print('--------------------------------------------------------')
        algo_rewards['myopic'] = baseline_myopic(rmab, init_states, budget=None)


        print('--------------------------------------------------------')
        print('run DQN')
        print('--------------------------------------------------------')

        mipper = Net2MIPPerScenario
        if not no_dqn: algo_rewards['iterative DQN'] = baseline_iterative_dqn(rmab, dqn_net, init_states)
        else: algo_rewards['iterative DQN'] = np.zeros(rmab.horizon * n_episodes_eval)
        if not no_dqn: algo_rewards['DQN MIP'] = MIP_results(rmab, dqn_net, mipper, init_states)
        else: algo_rewards['DQN MIP'] = np.zeros(rmab.horizon * n_episodes_eval)


        print('avg rewards (sem)')
        print(f'{rmab_type}  {rmab}')
        for algo in algo_rewards:
            print(f'  {algo.ljust(18, " ")}  {algo_rewards[algo].mean():.2f}, {stats.sem(algo_rewards[algo]):.2f}')

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        out_info = {
            'seed': args.seed,
            'rmab_type': rmab_type,
            'n_arms': rmab.n_arms,
            'n_samples': n_samples,
            'n_episodes_eval': n_episodes_eval,
            'horizon': horizon,
            'edge_prob': edge_prob,
            'start_time': start_time.strftime('%Y-%m-%d_%H-%M-%S'),
            'time': timestamp,
            'dqn_runtime': dqn_end_time - start_time,
        }
        for algo in algo_rewards:
            out_info[algo] = algo_rewards[algo].mean()

        
        if not no_dqn:
            df_out = pd.DataFrame([out_info])
            file_out = f'results_{prefix}{rmab}.csv'
            use_header = not os.path.exists(file_out)
            # append data frame to CSV file
            df_out.to_csv(file_out, mode='a', index=False, header=use_header)

        # plot per-timestep reward
        x_vals = np.arange(rmab.horizon * n_episodes_eval)
        plt.figure()
        for algo in algo_rewards:
            algo_rewards[algo].sort()
            plt.plot(x_vals, algo_rewards[algo], label=algo)
        plt.legend()
        plt.title(f'with N={rmab.action_dim} and J={rmab.n_arms}')
        plt.ylabel(f'Timestep ({n_episodes_eval} episodes, {rmab.horizon} horizon)')
        plt.ylabel('Per-timestep expected reward (sorted)')
        plt.tight_layout()
        # plt.show()

        # plot average reward
        plt.figure()
        bar_x = np.arange(len(algo_rewards))
        plt.bar(bar_x, [algo_rewards[algo].mean() for algo in algo_rewards], 
                yerr=[stats.sem(algo_rewards[algo]) for algo in algo_rewards], color='blue')
        plt.xticks(bar_x, [algo for algo in algo_rewards], rotation=30)
        plt.ylabel('Average expected reward')
        plt.xlabel('Method')
        plt.title(f'{rmab}')
        plt.tight_layout()
        
        plt.savefig(f'{out_dir}/{prefix}avg_reward_{rmab}_seed{args.seed}_{timestamp}.png')
        # plt.show()


        plt.close('all')
