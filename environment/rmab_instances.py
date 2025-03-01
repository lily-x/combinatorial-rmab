""" instantiate specific RMAB instances """

import random
import math
import numpy as np
import networkx as nx

from environment.graph_utils import visualize_graph
from environment.base_envs import StandardRMAB, TorchStandardRMAB, MultiStateRMAB, TorchMultiStateRMAB
from environment.scheduling import SchedulingRMAB, TorchSchedulingRMAB
from environment.constrained import ConstrainedRMAB, TorchConstrainedRMAB
from environment.routing import RoutingRMAB, TorchRoutingRMAB
from environment.multi_action import MultiActionRMAB, TorchMultiActionRMAB

import matplotlib.pyplot as plt
np.random.seed(42)


def get_wrapper(horizon, n_arms, budget, rmab_type='multistate', wrapper=None):
    # get underlying RMAB
    if rmab_type == 'standard':
        raise NotImplementedError
    elif rmab_type == 'multistate':
        rmab, t_rmab = multistate_rmab(horizon, n_arms=n_arms, budget=budget)

    # build wrapper class
    if wrapper == 'scheduling':
        n_timeslots = 5 #3
        wrapper_rmab = SchedulingRMAB(rmab, n_timeslots=n_timeslots)
        torch_wrapper_rmab = TorchSchedulingRMAB(t_rmab, n_timeslots=n_timeslots, arm_M=wrapper_rmab.arm_M, worker_M=wrapper_rmab.worker_M, arm_avail=wrapper_rmab.arm_avail, worker_avail=wrapper_rmab.worker_avail)
    elif wrapper == 'constrained':
        wrapper_rmab = ConstrainedRMAB(rmab)
        torch_wrapper_rmab = TorchConstrainedRMAB(t_rmab, wrapper_rmab.arm_costs, wrapper_rmab.worker_capacity)
    elif wrapper == 'routing':
        wrapper_rmab = RoutingRMAB(rmab)
        torch_wrapper_rmab = TorchRoutingRMAB(t_rmab, G=wrapper_rmab.G, G_pos=wrapper_rmab.G_pos, edges=wrapper_rmab.edges, edge_list=wrapper_rmab.edge_list, source=wrapper_rmab.source, valid_cycles=wrapper_rmab.valid_cycles)

    print('  base RMAB: ', rmab)
    print('  wrapper:   ', wrapper)
    return wrapper_rmab, torch_wrapper_rmab

def get_scheduling(horizon, n_arms, n_workers, rmab_type='standard'):
    print(f'making a scheduling problem with H{horizon}, J{n_arms}, workers{n_workers}')
    return get_wrapper(horizon, n_arms, n_workers, rmab_type=rmab_type, wrapper='scheduling')
    
def get_constrained(horizon, n_arms, n_workers, rmab_type='standard'):
    print(f'making a constrained problem with H{horizon}, J{n_arms}, workers{n_workers}')
    return get_wrapper(horizon, n_arms, n_workers, rmab_type=rmab_type, wrapper='constrained')
    
def get_routing(horizon, n_arms, budget, rmab_type='standard'):
    print(f'making a routing problem with H{horizon}, J{n_arms}, budget{budget}')
    return get_wrapper(horizon, n_arms, budget, rmab_type=rmab_type, wrapper='routing')


def multistate_rmab(horizon, n_arms=None, budget=None):
    """ create stylized two-arm, three-state setting where myopic fails """

    if n_arms is None:
        n_arms, budget = 2, 1

        transitions = np.array([[[0, 1], [-1, -1]],
                                [[0, 1], [-1, -1]]])
        state_r = np.array([[0.2, 0.1, 10],
                            [0.1, 0.2, 1]])
        init_state = np.zeros(n_arms)

        rmab = MultiStateRMAB(n_arms, budget, horizon, transitions, init_state=init_state, state_r=state_r)
        t_rmab = TorchMultiStateRMAB(n_arms, budget, horizon, transitions, init_state=rmab.init_state, state_r=rmab.state_r)

    else:
        # generic case where we create any number of arms
        bad_arms = np.random.choice(n_arms, budget, replace=False)

        ## make arms less extreme: 0 and 1
        transitions = np.ones((n_arms, 2, 2)) * -1
        transitions[:, 0, 0] = np.random.rand(n_arms) * 0.2
        transitions[:, 0, 1] = np.random.rand(n_arms) * 0.2 + 0.7

        init_state = np.zeros(n_arms)
        rmab = MultiStateRMAB(n_arms, budget, horizon, transitions, init_state=init_state, state_r=None)
        t_rmab = TorchMultiStateRMAB(n_arms, budget, horizon, transitions, init_state=rmab.init_state, state_r=rmab.state_r)

    print('transitions', transitions.shape)

    return rmab, t_rmab


def get_rmab_sigmoid(n_arms, action_dim, budget, horizon):
    prob_1 = 0.5 # probability of an arm starting in the good state
    init_state = np.random.choice([0, 1], size=(n_arms,), p=[1-prob_1, prob_1])

    # get multistate RMAB
    base_rmab, base_t_rmab = multistate_rmab(horizon, n_arms=n_arms, budget=budget)

    rmab = MultiActionRMAB(base_rmab, action_dim, init_state=init_state, link_type='sigmoid')
    t_rmab = TorchMultiActionRMAB(base_t_rmab, rmab)
    
    return rmab, t_rmab


