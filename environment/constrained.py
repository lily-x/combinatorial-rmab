"""
Capacity-constrained RMAB, in which each worker has their own budget constraint. 

Note: ConstrainedRMAB doesn't have a budget technically 
    (just depends on the workers' budget, and the cost of the arms)
"""

import random
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from environment.base_envs import StandardRMAB, TorchStandardRMAB
from approximator.constrained_approximator import ConstrainedRmabApproximator


class ConstrainedRMAB(StandardRMAB):
    def __init__(self, rmab, costs=None, capacity=None):
        self.n_workers = rmab.budget
        self.n_arms = rmab.n_arms
        self.rmab = rmab
        super().__init__(self.n_arms, rmab.budget, rmab.horizon, rmab.transitions, rmab.init_state)
        
        # set to none to ensure we're not accidentally using
        self.state = None
        self.transitions = None

        # create costs and capacity constraints
        if costs is None:
            self.arm_costs = np.random.choice(np.arange(2, 7), self.n_arms, replace=True)
            self.worker_capacity = np.random.choice(np.arange(2, 8), self.n_workers, replace=True)
        else:
            self.arm_costs = costs
            self.worker_capacity = capacity

        print(f'costs: {self.arm_costs}')
        print(f'capacity: {self.worker_capacity}')


    def __str__(self):
        return f'ConstrainedRMAB_{self.rmab}'
    
    def reset_to_state(self, state):
        return self.rmab.reset_to_state(state)
    
    def reset(self):
        return self.rmab.reset()
    
    def fresh_reset(self):
        return self.rmab.fresh_reset()

    def is_done(self):
        return self.rmab.is_done()

    def observation(self):
        return self.rmab.observation()
    
    def get_transitions(self):
        return self.rmab.get_transitions()
    
    def tup_to_action(self, tuples):
        assert len(tuples) == self.n_workers
        
        arm_actions = np.zeros(self.n_arms, dtype=np.float32)
        for i in range(self.n_workers):
            # ensure action is valid
            worker_costs = np.sum([self.arm_costs[j] for j in tuples[i]])
            assert worker_costs <= self.worker_capacity[i]

            for arm_j in tuples:
                arm_actions[arm_j] = 1

        return arm_actions


    def step(self, action, advance=True, allow_excess=True):
        """ an action has to be a set of tuples linking worker to arm """

        assert len(action) == self.n_arms

        # allow excess because we have different constraints here
        return self.rmab.step(action, advance=advance, allow_excess=True)


    def get_random_action(self):
        """ pick a random arm for each worker based on any constraints, ignoring overlaps """

        # walk down the workers, then pick all arms such that there's capacity
        workers = np.arange(self.n_workers)
        random.shuffle(workers)
        arms = np.arange(self.n_arms)
        random.shuffle(arms)
        
        assignment = [[] for i in range(self.n_workers)]
        for i in workers:
            remaining_cap = self.worker_capacity[i].item()
            for idx, arm_j in enumerate(arms):
                if arm_j == -1: continue    # if we've already  assigned this arm
                if self.arm_costs[arm_j] > remaining_cap: continue  # if we've already exceeded budget
                
                assignment[i].append(arm_j)
                remaining_cap -= self.arm_costs[arm_j]
                arms[idx] = -1

        action = self.tup_to_action(assignment)
        return action
    
    def get_null_action(self):
        return self.rmab.get_null_action()
    
    def calc_action_expected_value(self, action):
        return self.rmab.calc_action_expected_value(action)

    def get_approximator(self):
        """ get MIP approximator with action constraints """
        return ConstrainedRmabApproximator
            

class TorchConstrainedRMAB(ConstrainedRMAB):
    """ torch version of ConstrainedRMAB """
    def __init__(self, rmab, costs=None, capacity=None):
        super().__init__(rmab, costs=costs, capacity=capacity)

        if costs is None:
            self.arm_costs = torch.tensor(self.arm_costs, dtype=torch.float32, requires_grad=False)
        else:
            self.arm_costs = costs
        if capacity is None:
            self.worker_capacity = torch.tensor(self.worker_capacity, dtype=torch.float32, requires_grad=False)
        else:
            self.worker_capacity = capacity
        

    
    def __str__(self):
        return f'Torch{super().__str__()}'
    
    def step(self, action, advance=True, allow_excess=True):
        # allow excess because we have different constraints here
        return self.rmab.step(action, advance=advance, allow_excess=True)

    def get_random_action(self):
        random_action = super().get_random_action()
        return torch.tensor(random_action, requires_grad=False).float()
    
    def get_null_action(self):
        return torch.zeros(self.n_arms)
    
    def calc_action_expected_value(self, action):
        return self.rmab.calc_action_expected_value(action)

    def get_constraints(self, mip):
        """ given a gurobi model, add action feasibility constraints """
        raise NotImplementedError