"""
implement baselines
"""

import numpy as np
import random

from environment.base_envs import StandardRMAB, MultiStateRMAB
from environment.multi_action import MultiActionRMAB
from environment.routing import RoutingRMAB
from environment.constrained import ConstrainedRMAB
from environment.scheduling import SchedulingRMAB
from environment.scheduling import get_items_in_common


def baseline_optimal(rmab, init_states):
    """ Act on the arms with highest end-state reward at each timestep
    NOTE that this is just a metric. It doesn't consider probabilities of transitioning
    This also doesn't respect any constraint except the budget constraint 
    e.g., it ignores all combinatorial constraints """
    n_episodes = len(init_states)
    rewards = np.zeros(n_episodes * rmab.horizon)
    if not isinstance(rmab, MultiStateRMAB): 
        rmab = rmab.rmab # get inner RMAB
        if not isinstance(rmab, MultiStateRMAB): return rewards 
    

    for ep in range(n_episodes):
        rmab.reset_to_state(init_states[ep])
        for t in range(rmab.horizon):
            end_rewards = rmab.state_r
            
            best_rewards = np.argsort(end_rewards)[::-1]
            top_actions = best_rewards[:rmab.budget]
            action = np.zeros(rmab.action_dim, dtype=np.int32)
            action[top_actions] = 1

            _, reward, _, _, info = rmab.step(action)
            rewards[ep*rmab.horizon + t] = info['next_state_expected']

    return rewards


def optimize_myopic(rmab):
    """ solve the MIP using a myopic action, only looking at reward at next timestep """
    if not (isinstance(rmab, StandardRMAB) or isinstance(rmab, MultiActionRMAB) or isinstance(rmab, MultiStateRMAB)):
        raise NotImplementedError
    
    approximator = rmab.get_approximator()(rmab)
    results = approximator.optimize_myopic()
    
    action = results['sol'].astype(int)
    return action


def baseline_null_action(rmab, init_states):
    """ take no action at each timestep """
    n_episodes = len(init_states)
    rewards = np.zeros(n_episodes * rmab.horizon)
    for ep in range(n_episodes):
        rmab.reset_to_state(init_states[ep])
        for t in range(rmab.horizon):
            null_action = rmab.get_null_action()

            _, reward, _, _, info = rmab.step(null_action)
            rewards[ep*rmab.horizon + t] = info['next_state_expected']

    return rewards


def baseline_random(rmab, init_states):
    """ take a random action at each timestep
    randomly pick an action within constraints """
    n_episodes = len(init_states)
    rewards = np.zeros(n_episodes * rmab.horizon)
    for ep in range(n_episodes):
        rmab.reset_to_state(init_states[ep])
        for t in range(rmab.horizon):
            rand_action = rmab.get_random_action()

            _, reward, _, _, info = rmab.step(rand_action)
            rewards[ep*rmab.horizon + t] = info['next_state_expected']

    return rewards


def baseline_sample(rmab, init_states, n_samples=20):
    """ sample a bunch of actions and estimate the reward for each """
    n_episodes = len(init_states)
    rewards = np.zeros(n_episodes * rmab.horizon)
    for ep in range(n_episodes):
        rmab.reset_to_state(init_states[ep])
        for t in range(rmab.horizon):
            # general pool of random actions within budget constraint
            random_action_pool = [rmab.get_random_action() for _ in range(n_samples)]

            best_action = None
            best_val = -1
            for rand_action in random_action_pool:
                # expected reward
                expected_val = rmab.calc_action_expected_value(rand_action)

                if expected_val > best_val:
                    best_action = rand_action
                    best_val = expected_val

            _, reward, _, _, info = rmab.step(best_action)
            rewards[ep*rmab.horizon + t] = info['next_state_expected']

    return rewards


def baseline_myopic(rmab, init_states, budget=None):
    """ myopically take an action that maximizes the reward at the next timestep 
    greedy approach that just myopically maximizes the one-step best solution approach 
    
    allow budget to be selected by user for faster customization"""

    # if budget selected, override RMAB budget
    if budget is None:
        budget = rmab.budget

    n_episodes = len(init_states)
    rewards = np.zeros(n_episodes * rmab.horizon)
    for ep in range(n_episodes):
        rmab.reset_to_state(init_states[ep])
        for t in range(rmab.horizon):
            action = optimize_myopic(rmab) 
            _, reward, _, _, info = rmab.step(action)
            rewards[ep*rmab.horizon + t] = info['next_state_expected']

    return rewards


def baseline_greedy_iterative_myopic(rmab, init_states, budget=None):
    """ iterative solution approach, where we greedily pick the next action that maximizes myopic reward """

    print('------------------------------------')
    print('greedy, iterative, myopic baseline')
    print('------------------------------------')

    n_episodes = len(init_states)

    # if budget selected, override RMAB budget
    if budget is None:
        budget = rmab.budget

    rewards = np.zeros(n_episodes * rmab.horizon)
    for ep in range(n_episodes):
        rmab.reset_to_state(init_states[ep])
        for t in range(rmab.horizon):
            iterative_actions = np.zeros(rmab.action_dim, dtype=np.int32)  # iteratively build up actions
            
            curr_reward = rmab.calc_action_expected_value(iterative_actions)
            curr_budget = budget


            if isinstance(rmab, MultiActionRMAB):  #  or isinstance(rmab, MultiStateRMAB)
                while curr_budget > 0:
                    best_action = None
                    best_reward = -1
                    
                    # pick action that has the best immediate reward improvement
                    for i in range(rmab.action_dim):
                        if iterative_actions[i] == 1: # if action has already been selected
                            continue

                        try_action = iterative_actions.copy()
                        try_action[i] = 1

                        action_reward = rmab.calc_action_expected_value(try_action)
                        if action_reward > best_reward:
                            best_reward = action_reward
                            best_action = i

                    if best_reward <= curr_reward:  # if no actions improve reward, skip
                        break
                        
                    if best_action is not None:
                        iterative_actions[best_action] = 1
                        curr_reward = best_reward
                        curr_budget -= 1


            elif isinstance(rmab, SchedulingRMAB):
                # sort arms by expected value
                expected_value = np.zeros(rmab.n_arms)
                try_actions = np.eye(rmab.n_arms, dtype=np.int32)
                for j in range(rmab.n_arms):
                    try_value = rmab.calc_action_expected_value(try_actions[j])
                    expected_value[j] = try_value
                arms = np.argsort(expected_value)[::-1]

                workers = np.arange(rmab.n_workers)
                random.shuffle(workers)

                # list of available timeslots per worker
                available_workers = {}
                for i in range(rmab.n_workers):
                    available_workers[i] = rmab.worker_avail[i].copy()
                
                # for each arm, find two workers both with compatible slots
                for j in arms:
                    not_set = True
                    if iterative_actions.sum() >= rmab.budget: break
                    for i1 in workers:
                        if not_set:
                            compatible1, slots1 = get_items_in_common(available_workers[i1], rmab.arm_avail[j])
                            if not compatible1: continue
                            for i2 in workers:
                                if i1 == i2: continue
                                compatible2, slots2 = get_items_in_common(available_workers[i2], rmab.arm_avail[j])
                                if not compatible2: continue

                                available_workers[i1].remove(slots1.pop())
                                available_workers[i2].remove(slots2.pop())
                                iterative_actions[j] = 1
                                not_set = False
                                break


            elif isinstance(rmab, ConstrainedRMAB):
                assert rmab.action_dim == rmab.n_arms
                
                # arrange workers from lowest capacity to highest capacity
                workers = np.argsort(rmab.worker_capacity)

                # walk down the workers, then pick all arms such that there's capacity
                available_arm_costs = rmab.arm_costs.copy()
                
                for i in workers:
                    best_value = -1
                    best_arm = None
                    remaining_cap = rmab.worker_capacity[i].item()
                    lowest_cost = np.min(available_arm_costs[available_arm_costs != -1])  # find cheapest available arm
                    while remaining_cap > lowest_cost:  # while we can still afford any arm
                        for j in range(rmab.n_arms):
                            if available_arm_costs[j] == -1: continue   # if we've already assigned this arm
                            if available_arm_costs[j] > remaining_cap: continue  # if we've already exceeded budget

                            try_action = iterative_actions.copy()
                            try_action[j] = 1

                            try_value = rmab.calc_action_expected_value(try_action)
                            try_benefit = (try_value - curr_reward) / available_arm_costs[j]  # find the best cost ratio
                            if try_benefit < 0: continue  # make sure that we're increasing value
                            
                            if try_benefit > best_value:
                                best_value = try_benefit
                                best_arm = j
                        
                        if best_arm is not None:
                            available_arm_costs[best_arm] = -1
                            remaining_cap -= rmab.arm_costs[best_arm]
                            iterative_actions[best_arm] = 1

                            lowest_cost = np.min(available_arm_costs[available_arm_costs != -1])  # find cheapest available arm

            elif isinstance(rmab, RoutingRMAB):
                return rewards  # just return all 0s; skip this baseline
            
            else:
                raise ValueError(f'Unsupported RMAB type: {type(rmab)}')
            
            _, reward, _, _, info = rmab.step(iterative_actions)
            rewards[ep*rmab.horizon + t] = info['next_state_expected']

    return rewards

