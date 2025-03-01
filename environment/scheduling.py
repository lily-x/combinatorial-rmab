"""
Schedule-constrained RMAB, in which each worker has their own schedule with limited availability,
and each patient also has their own availability. 

Similar to bipartite matching.
"""

import random
import numpy as np
import gym

import gurobipy as gp
from gurobipy import GRB

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from environment.base_envs import StandardRMAB, TorchStandardRMAB
from approximator.scheduling_approximator import SchedulingRmabApproximator


def item_in_common(a, b):
    """ return whether lists a and b have any items in common """
    return any(x in set(b) for x in a)


def get_items_in_common(a, b):
    shared_items = set(a).intersection(b)
    if len(shared_items) == 0:
        return False, None
    else:
        return True, shared_items


class SchedulingRMAB(StandardRMAB):
    def __init__(self, rmab, n_timeslots, arm_M=None, worker_M=None, arm_avail=None, worker_avail=None):
        assert n_timeslots >= 3
        
        # use a specific underlying RMAB - to use with multistate RMAB
        self.n_workers = rmab.budget
        self.n_arms = rmab.n_arms
        self.rmab = rmab

        super().__init__(self.n_arms, rmab.budget, rmab.horizon, rmab.transitions, rmab.init_state)

        self.action_dim = rmab.n_arms
        self.n_timeslots = n_timeslots
        
        # set to none to ensure we're not accidentally using
        self.state = None
        self.transitions = None

        # below are all scheduling-specific things
        timeslots = np.arange(self.n_timeslots)


        # generate list of available timeslots
        if arm_avail is None:
            # availability matrices
            arm_M = np.zeros((self.n_arms, n_timeslots))
            worker_M = np.zeros((self.n_arms, n_timeslots))

            arm_avail = {}
            worker_avail = {}

            # set availability probabilities
            arm_avail_prob = np.sort(np.random.rand(n_timeslots))
            worker_avail_prob = np.sort(np.random.rand(n_timeslots))[::-1]
            arm_avail_prob = arm_avail_prob / np.sum(arm_avail_prob)  # normalize
            worker_avail_prob = worker_avail_prob / np.sum(worker_avail_prob)

            for j in range(self.n_arms):
                n_available = 2 # np.max([2, np.ceil(n_timeslots/3).astype(int)]) # 2
                available_timeslots = np.random.choice(timeslots, n_available, p=arm_avail_prob, replace=False)
                arm_avail[j] = available_timeslots.tolist()
                arm_M[j, available_timeslots] = 1

            for i in range(self.n_workers):
                n_available = 3 # np.max([2, np.ceil(n_timeslots/3).astype(int)])
                available_timeslots = np.random.choice(timeslots, n_available, p=worker_avail_prob, replace=False)
                
                worker_avail[i] = available_timeslots.tolist()
                worker_M[i, available_timeslots] = 1

        self.arm_avail = arm_avail
        self.worker_avail = worker_avail
        self.arm_M = arm_M
        self.worker_M = worker_M

        print(f'arm avail:    {self.arm_avail}')
        print(f'worker avail: {self.worker_avail}')
            

    def __str__(self):
        return f'SchedulingRMAB_n_workers{self.n_workers}_n_timeslots{self.n_timeslots}_{self.rmab}'

    def worker_compatible(self, i, j):
        """ return whether worker i is compatible with arm j """
        return item_in_common(self.worker_avail[i], self.arm_avail[j])
    
    def get_compatible_slots(self, i, j):
        return get_items_in_common(self.worker_avail[i], self.arm_avail[j])

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

    def step(self, action, advance=True, allow_excess=False):
        """ an action is a list of arms acted on
        (note that here we're not checking to ensure valid action) """
        assert len(action) == self.n_arms
        return self.rmab.step(action, advance=advance, allow_excess=allow_excess)


    def get_random_action(self):
        """ pick a random arm for each worker based on any scheduling constraints, ignoring overlaps """

        # create dictionary of how many workers are available for the different timeslots
        action = np.zeros(self.n_arms, dtype=np.float32)

        # walk down the workers, then pick the first arm that there's a compatible timestep with
        workers = np.arange(self.n_workers)
        arms = np.arange(self.n_arms)
        random.shuffle(workers)
        random.shuffle(arms)

        # list of available timeslots per worker
        available_workers = {}
        for i in range(self.n_workers):
            available_workers[i] = self.worker_avail[i].copy()
        
        # for each arm, find two workers both with compatible slots
        for j in arms:
            not_set = True
            if action.sum() >= self.budget: break
            for i1 in workers:
                if not_set:
                    compatible1, slots1 = get_items_in_common(available_workers[i1], self.arm_avail[j])
                    if not compatible1: continue
                    for i2 in workers:
                        if i1 == i2: continue
                        compatible2, slots2 = get_items_in_common(available_workers[i2], self.arm_avail[j])
                        if not compatible2: continue
                        
                        available_workers[i1].remove(slots1.pop())    
                        available_workers[i2].remove(slots2.pop())
                        action[j] = 1
                        not_set = False
                        break
                

        return action
    
    def get_null_action(self):
        return self.rmab.get_null_action()
    
    def calc_action_expected_value(self, action):
        return self.rmab.calc_action_expected_value(action)

    def get_approximator(self):
        """ get MIP approximator with action constraints """
        return SchedulingRmabApproximator
            

class TorchSchedulingRMAB(SchedulingRMAB):
    """ torch version of SchedulingRMAB """
    def __init__(self, rmab, n_timeslots, arm_M=None, worker_M=None, arm_avail=None, worker_avail=None):
        super().__init__(rmab, n_timeslots=n_timeslots, arm_M=arm_M, worker_M=worker_M, arm_avail=arm_avail, worker_avail=worker_avail)

        self.arm_M = torch.tensor(self.arm_M, dtype=torch.float32, requires_grad=False)
        self.worker_M = torch.tensor(self.worker_M, dtype=torch.float32, requires_grad=False)

    def __str__(self):
        return f'Torch{super().__str__()}'
    
    def step(self, action, advance=True, allow_excess=False):
        return self.rmab.step(action, advance=advance, allow_excess=allow_excess)
    
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