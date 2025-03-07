"""
path routing problem over a standard RMAB problem
"""

import numpy as np

import networkx as nx

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from environment.base_envs import StandardRMAB
from approximator.routing_approximator import RoutingRmabApproximator

import matplotlib.pyplot as plt
import random
import pickle


class RoutingRMAB(StandardRMAB):
    def __init__(self, rmab, G=None, G_pos=None, edges=None, edge_list=None, source=None, valid_cycles=None):
        # for the routing problem, we set the max path length as double the budget
        # (but we can still only pull "budget" total of these arms that we visit)
        self.budget = rmab.budget
        self.max_path_length = 2 * rmab.budget
        self.n_arms = rmab.n_arms
        self.rmab = rmab
        super().__init__(self.n_arms, rmab.budget, rmab.horizon, rmab.transitions, rmab.init_state)

        self.action_dim = rmab.n_arms

        # set to none to ensure we're not accidentally using
        self.state = None
        self.transitions = None

        if G is None:
            assert self.n_arms in [20, 40, 60, 80, 100]
            in_file ='./environment/network/tube_network.pkl'
            with open(in_file, 'rb') as handle:
                tube_network = pickle.load(handle)
            G = tube_network[f'G[{self.n_arms}]']
            G_pos = nx.spring_layout(G)
            
            edges = nx.to_numpy_array(G)  # [N x N] matrix: position 0 indicates the source/sink node
            edge_list = [(j,k) for j in range(self.n_arms) for k in range(self.n_arms) if edges[j,k] == 1]

            if self.n_arms == 20: source = 0
            elif self.n_arms == 40: source = 10
            elif self.n_arms == 60: source = 20
            elif self.n_arms == 80: source = 0
            elif self.n_arms == 100: source = 0
            else: raise Exception('source node not defined for this number of arms')

            #### generate random actions
            # generate all simple cycles within path_length
            random_path_length = min(self.max_path_length, self.n_arms)
            cycles = nx.simple_cycles(G, length_bound=random_path_length)

            valid_cycles = []
            # find all simple cycles containing source
            for cycle in cycles:
                if len(cycle) < 2:
                    continue
                if source in cycle:
                    valid_cycles.append(cycle)

            print(f'there are {len(valid_cycles)} random actions')

        self.G = G
        self.G_pos = G_pos
        self.edges = edges
        self.edge_list = edge_list
        self.source = source
        self.valid_cycles = valid_cycles
        
        self.plot_graph(title=f'tube_network N{self.n_arms}', show=False)


    def plot_graph(self, path=None, action=None, title='', show=True):
        plt.figure(figsize=(8,5))
        nx.draw_networkx(self.G, self.G_pos, node_color='lightblue', 
                with_labels=True, 
                node_size=500)
    
        if path is not None:
            if isinstance(path, torch.Tensor):
                path = path.int().tolist()
            elif isinstance(path, np.ndarray):
                path = path.astype(int).tolist()

            path = path + [self.source]
            nx.draw_networkx_nodes(self.G, self.G_pos, nodelist=path, node_color='r', node_size=550)
            nx.draw_networkx_edges(self.G, self.G_pos, edgelist=[(path[i], path[i+1]) for i in range(len(path)-1)], edge_color='r', width=2)

        if action is not None:
            nx.draw_networkx_nodes(self.G, self.G_pos, nodelist=path, node_color='r', node_size=550)

        if path is not None:
            title = f'{title}_max_path{self.self.max_path_length}_path{path}'
        if action is not None:
            title = f'{title}_max_path{self.self.max_path_length}_action{action}'
        
        plt.title(title)
        plt.savefig(f'graph_n{self.n_arms}_{title}.png')
        if show: plt.show()
        else: plt.close()

    def has_edge(self, j, k):
        return self.edges[j][k] == 1

    def path_to_action(self, path):
        """ an path is a set of nodes to follow; convert to binary action vector """
        # convert path to action
        action = np.zeros(self.n_arms, dtype=np.int32)
        if isinstance(path, torch.Tensor):
            path = path.int().tolist()
        elif isinstance(path, np.ndarray):
            path = path.astype(int).tolist()

        for node in path:
            action[node] = 1
        return action

    def __str__(self):
        return f'RoutingRMAB_path{self.max_path_length}_{self.rmab}'

    def reset_to_state(self, state):
        return self.rmab.reset_to_state(state)
    
    def reset(self):
        return self.rmab.reset()

    def is_done(self):
        return self.rmab.is_done()
    
    def fresh_reset(self):
        return self.rmab.fresh_reset()
    
    def observation(self):
        return self.rmab.observation()
    
    def get_transitions(self):
        return self.rmab.get_transitions()

    def step(self, action, advance=True, allow_excess=False):
        assert len(action) == self.n_arms
        assert action.sum() <= self.budget
        return self.rmab.step(action, advance=advance, allow_excess=allow_excess)

    def get_random_action(self, plot=False):
        """ pick a random feasible path """
        # pick random item in self.valid_cycles
        path = random.choice(self.valid_cycles)

        # pick a valid subset of nodes to pull
        selected_nodes = np.random.choice(path, size=min(self.budget, len(path)), replace=False)

        if plot:
            self.plot_graph(path=path, action=None, title='random_action', show=True)

        return self.path_to_action(selected_nodes)

    def get_null_action(self):
        return self.rmab.get_null_action()
    
    def calc_action_expected_value(self, action):
        return self.rmab.calc_action_expected_value(action)
            
    def get_approximator(self):
        """ get MIP approximator with action constraints """
        return RoutingRmabApproximator
            


class TorchRoutingRMAB(RoutingRMAB):
    """ torch version of RoutingRMAB """
    def __init__(self, rmab, G=None, G_pos=None, edges=None, edge_list=None, source=None, valid_cycles=None):
        super().__init__(rmab, G=G, G_pos=G_pos, edges=edges, edge_list=edge_list, source=source, valid_cycles=valid_cycles)

        self.edges = torch.tensor(self.edges, dtype=torch.float32, requires_grad=False)
        self.valid_cycles = [torch.tensor(cycle, dtype=torch.float32, requires_grad=False) for cycle in self.valid_cycles]

        if self.rmab.transitions is None:
            print('no transitions!')
            import pdb; pdb.set_trace()

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
    