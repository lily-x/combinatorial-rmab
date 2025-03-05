""" implement the iterative DQN baseline:
an iterative approach that leverages the trained Q-network 
used by the DQN-MIP solver but, at inference time,
uses an iterative solution rather than solving with the MIP """

import torch
import numpy as np
import random

from environment.multi_action import MultiActionRMAB
from environment.routing import RoutingRMAB
from environment.constrained import ConstrainedRMAB
from environment.scheduling import SchedulingRMAB
from environment.scheduling import get_items_in_common


def dqn_estimate_action_value(rmab, net, action):
    """ action can be a single action (1d array) or a batch of actions (2d array) """

    state = rmab.observation()

    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).float()

    # convert to right shape if it's a single item
    if len(action.shape) == 1:
        action = action.unsqueeze(dim=0)
    if len(state.shape) == 1:
        state = state.unsqueeze(dim=0)
        state = state.repeat(action.shape[0], 1)
    
    # compute Q(s_t, a)
    input = torch.concatenate([action, state], axis=1)
    
    # NOTE: negating (-) because DQN is set to minimize
    state_action_values = -net(input)  #.gather(1, action_batch)

    return state_action_values


def baseline_iterative_dqn(rmab, net, init_states, budget=None):
    print('------------------------------------')
    print('greedy, iterative DQN baseline')
    print('------------------------------------')

    n_episodes = len(init_states)

    # if budget selected, override RMAB budget
    if budget is None:
        budget = rmab.budget

    values = np.zeros(n_episodes * rmab.horizon)
    for ep in range(n_episodes):
        rmab.reset_to_state(init_states[ep])
        for t in range(rmab.horizon):
            curr_action = torch.zeros(rmab.action_dim, dtype=torch.int32, requires_grad=False)
            curr_value = dqn_estimate_action_value(rmab, net, curr_action).item()
            curr_budget = budget

            if isinstance(rmab, MultiActionRMAB):
                # identity matrix enables us to try adding each action
                identity = torch.eye(rmab.action_dim, dtype=torch.int32, requires_grad=False)
                while curr_budget > 0:
                    curr_action_repeat = curr_action.unsqueeze(dim=0).repeat(rmab.action_dim, 1)
                    try_actions = torch.maximum(curr_action_repeat, identity)  # take max to not exceed 1

                    # evaluate all actions together in a single forward pass
                    action_values = dqn_estimate_action_value(rmab, net, try_actions)
                    best_action = torch.argmax(action_values)

                    if action_values[best_action].item() <= curr_value:  # if no actions improve reward, skip
                        break

                    curr_action = try_actions[best_action]
                    curr_value = action_values[best_action].item()
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
                    if curr_action.sum() >= rmab.budget: break
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
                                curr_action[j] = 1
                                not_set = False
                                break


            elif isinstance(rmab, ConstrainedRMAB):
                assert rmab.action_dim == rmab.n_arms
                
                # arrange workers from lowest capacity to highest capacity
                workers = np.argsort(rmab.worker_capacity)

                # walk down the workers, then pick all arms such that there's capacity
                available_arm_costs = rmab.arm_costs.copy()
                
                for i in workers:
                    remaining_cap = rmab.worker_capacity[i].item()
                    lowest_cost = np.min(available_arm_costs[available_arm_costs != -1])  # find cheapest available arm
                    while remaining_cap > lowest_cost:  # while we can still afford any arm
                        best_value = -1
                        best_arm = None
                        for j in range(rmab.n_arms):
                            if available_arm_costs[j] == -1: continue   # if we've already assigned this arm
                            if available_arm_costs[j] > remaining_cap: continue  # if we've already exceeded budget

                            try_action = curr_action.detach().clone()
                            try_action[j] = 1

                            try_value = dqn_estimate_action_value(rmab, net, try_action).item()
                            try_benefit = (try_value - curr_value) / available_arm_costs[j]  # find the best cost ratio
                            if try_benefit < 0: continue  # make sure that we're increasing value
                            
                            if try_benefit > best_value:
                                best_value = try_benefit
                                best_arm = j
                        
                        if best_arm is not None:
                            available_arm_costs[best_arm] = -1
                            remaining_cap -= rmab.arm_costs[best_arm]
                            curr_action[best_arm] = 1

                        lowest_cost = np.min(available_arm_costs[available_arm_costs != -1])  # find cheapest available arm

            elif isinstance(rmab, RoutingRMAB):
                return values  # just return all 0s; skip this baseline

            else:
                raise ValueError(f'Unsupported RMAB type: {type(rmab)}')

            curr_action = curr_action.numpy()
            # print('    selected action:', curr_action)
            _, reward, _, _, info = rmab.step(curr_action)
            values[ep*rmab.horizon + t] = info['next_state_expected']

    return values
