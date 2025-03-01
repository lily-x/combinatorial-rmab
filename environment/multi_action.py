"""
multi-action with sigmoid link function

allows for different sigmoid parameters per arm
"""

import random
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from environment.graph_utils import make_geometric, num_to_chr, visualize_graph
from environment.base_envs import BaseRMAB, MultiStateRMAB

from approximator.multi_action_rmab_approximator import MultiActionRmabApproximator


##############################################################
# multi-action RMAB
##############################################################

class MultiActionRMAB(BaseRMAB):
    """ binary-action RMAB with multiple actions """
    def __init__(self, rmab, action_dim, init_state=None, link_type='sigmoid', make_weights=True):
        if not link_type == 'sigmoid': raise NotImplementedError
        assert rmab.budget < action_dim

        self.rmab = rmab
        self.action_dim = action_dim

        self.n_arms = rmab.n_arms
        self.budget = rmab.budget
        self.horizon = rmab.horizon
        self.link_type = link_type

        assert isinstance(rmab, MultiStateRMAB)

        # set to none to ensure we're not accidentally using
        self.state = None
        self.transitions = None

        super().__init__(self.n_arms, self.action_dim, self.budget, self.horizon, init_state)

        # position of nodes for networkx visualization
        self.node_pos = {}
        for action in range(self.action_dim):
            self.node_pos[action] = (0, -action - 0.5)
        for arm in range(self.n_arms):
            arm_label = num_to_chr(arm)
            self.node_pos[arm_label] = (3, -arm)

        self.good_arms = np.arange(np.ceil(self.n_arms * (self.budget / self.action_dim)).astype(int), self.n_arms)  # pick the top arms, which have the highest probability of good transition
        self.good_actions = np.random.choice(self.action_dim, self.budget, replace=False)

        self.bad_arms = [j for j in range(self.n_arms) if j not in self.good_arms]
        self.bad_actions = [i for i in range(self.action_dim) if i not in self.good_actions]
        print('good arms', self.good_arms)
        print('bad arms', self.bad_arms)
        print('good actions', self.good_actions)
        print('bad actions', self.bad_actions)
        self.a_vals = np.zeros(self.n_arms)
        self.b_vals = np.zeros(self.n_arms)
        self.x0_vals = np.zeros(self.n_arms)

        # parameters for the sigmoid function of each arm
        for j in self.good_arms:
            self.a_vals[j] = (np.random.rand(1) * .3 + .7).round(1)
            self.b_vals[j] = (np.random.rand(1) * .5 + 1.4).round(1)
            self.x0_vals[j] = np.random.randint(9, 11)
        for j in self.bad_arms:
            self.a_vals[j] = (np.random.rand(1) * .3 + .2).round(1)
            self.x0_vals[j] = np.random.randint(1, 3)
            self.b_vals[j] = 1

        print('a vals', self.a_vals)
        print('b vals', self.b_vals)
        print('x0 vals', self.x0_vals)

        self.p_change = 3      # probability gain from an action

        if make_weights:
            if link_type == 'linear':
                self.get_linear_weights()
                # for linear, ensure we never exceed 1 (total probability)
                assert np.all(self.weights_p.sum(axis=1) + self.theta_p < 1)

            elif link_type == 'sigmoid':
                self.get_sigmoid_weights()

                # plot bipartite graph and weights
                out_dir = './'
                visualize_graph(self, out_dir, good_actions=self.good_actions, bad_actions=self.bad_actions)

            else: raise NotImplementedError

            assert len(self.theta_p) == self.n_arms
            assert self.weights_p.shape == (self.n_arms, self.action_dim)

        plot_sigmoid = False
        if plot_sigmoid:
            plt.figure()
            x_vals = np.linspace(0, 10, 100)
            for j in self.good_arms:
                y_vals = self.link_function(j, x_vals)
                plt.plot(x_vals, y_vals, 'g', label=f'arm {j} a{self.a_vals[j]} b{self.b_vals[j]} x0{self.x0_vals[j]}')
            for j in self.bad_arms:
                y_vals = self.link_function(j, x_vals)
                plt.plot(x_vals, y_vals, 'b-.', label=f'arm {j} a{self.a_vals[j]} b{self.b_vals[j]} x0{self.x0_vals[j]:.0f}')
            # plt.legend()
            plt.xlabel('Input $\omega^T \mathbf{a}$')
            plt.ylabel('Output from sigmoid link function $\phi$')
            plt.savefig(f'sigmoid_functions_N{rmab.n_arms}.png', bbox_inches='tight')
            plt.show()

    def __str__(self):
        return f'MultiActionRMAB_{self.link_type}_N{self.action_dim}_J{self.n_arms}_B{self.budget}_H{self.horizon}'
    
    def observation(self):
        return self.rmab.observation()

    def advance(self):
        self.rmab.advance()

    def is_done(self):
        return self.rmab.is_done()
    
    def reset(self):
        return self.rmab.reset()
    
    def fresh_reset(self):
        return self.rmab.fresh_reset()
    
    def set_state(self, state):
        return self.rmab.set_state(state)


    def get_sigmoid_weights(self):
        """ randomly generate weights \omega for the sigmoid link function """

        assert self.link_type == 'sigmoid'

        self.weights_p = np.zeros((self.n_arms, self.action_dim))

        # base probability
        self.theta_p = np.zeros(self.n_arms)

        modulo = 3 # moduolo value - how many actions to assign to the same arm
        n_arms_to_act = 3

        for idx, action_i in enumerate(self.good_actions):
            if idx % modulo == 0 and idx <= len(self.good_actions) - modulo:  # if even or the last item 
                not_acted_on = [i for i in self.good_arms if self.weights_p[i, :].sum() == 0]
                selected_arms = np.random.choice(not_acted_on, n_arms_to_act, replace=False)
                self.weights_p[selected_arms, action_i] = self.p_change
                for mm in range(modulo):
                    self.weights_p[selected_arms, self.good_actions[idx+mm]] = self.p_change
            elif idx >= len(self.good_actions) - (len(self.good_actions) % modulo):
                not_acted_on = [i for i in self.good_arms if self.weights_p[i, :].sum() <= 1]
                selected_arms = np.random.choice(not_acted_on, n_arms_to_act, replace=False)
                self.weights_p[selected_arms, action_i] = self.p_change

        for idx, action_i in enumerate(self.bad_actions):
            if idx % modulo == 0 and idx <= len(self.bad_actions) - modulo:  # if even or the last item 
                not_acted_on = [i for i in self.bad_arms if self.weights_p[i, :].sum() == 0]
                selected_arms = np.random.choice(not_acted_on, n_arms_to_act, replace=False)
                self.weights_p[selected_arms, action_i] = self.p_change
                for mm in range(modulo):
                    self.weights_p[selected_arms, self.bad_actions[idx+mm]] = self.p_change
            elif idx >= len(self.bad_actions) - (len(self.bad_actions) % modulo):
                not_acted_on = [i for i in self.bad_arms if self.weights_p[i, :].sum() <= 1]
                selected_arms = np.random.choice(not_acted_on, n_arms_to_act, replace=False)
                self.weights_p[selected_arms, action_i] = self.p_change


    def get_linear_weights(self):
        """ randomly generate weights for linear link function """
        assert self.link_type == 'linear'
        rand_values_p = np.random.random((self.n_arms, self.action_dim+1))
        rand_values_q = np.random.random((self.n_arms, self.action_dim+1))

        # pick which edges to keep (by making many of them 0)
        for i in range(self.action_dim):
            for j in range(self.n_arms):
                edge_sample = random.random()
                # print('  edge_sample', edge_sample)
                if edge_sample > self.edge_prob:
                    rand_values_p[j,i+1] = 0
                    rand_values_q[j,i+1] = 0

        generator_method = 'helpful_arms'

        if generator_method == 'helpful_arms':
            # pick subset of arms where acting helps
            helpful_arms = np.random.choice(self.n_arms, size=self.budget, replace=False)
            unhelpful_arms = np.array(list(set(np.arange(self.n_arms)) - set(helpful_arms)))

            total_prob = np.zeros((self.n_arms, 2))
            total_prob[unhelpful_arms, :] = np.random.uniform(0.1, 0.3, size=(self.n_arms-self.budget, 2))
            total_prob[helpful_arms, :] = np.random.uniform(0.5, 0.8, size=(self.budget, 2))


        elif generator_method == 'random_prob':
            # pick a max probability for each arm
            total_prob = np.random.uniform(0.1, 1, size=(self.n_arms, 2))

        else:
            raise NotImplementedError
        
        total_p = total_prob.min(axis=1)

        # scale all values down to keep valid probabilities [0, 1] 
        rand_values_p = rand_values_p.T / rand_values_p.sum(axis=1)

        # scale values down so they don't all sum to 1
        rand_values_p = (total_p * rand_values_p).T

        self.theta_p = rand_values_p[:, 0]
        
        self.weights_p = rand_values_p[:, 1:]


    def get_random_action(self):
        """ randomly pick an action within constraints """
        selected_actions = np.random.choice(self.action_dim, self.budget, replace=False)
        random_action = np.zeros(self.action_dim, dtype=np.float32)
        random_action[selected_actions] = 1

        return random_action
    

    def get_null_action(self):
        """ no action """
        return np.zeros(self.action_dim, dtype=np.float32)


    def step(self, action, advance=True, allow_excess=False):
        """ allow for multiple actions to be taken simultaneously """
        if isinstance(action, list):
            action = np.array(action)
        assert len(action) == self.action_dim

        # check budget constraint
        if not allow_excess: assert action.sum() <= self.budget + 1e-5, f'action is {action} which is more than budget {self.budget}'
        
        state = self.observation()
        
        # get a transition probability based on the implemented action
        # dependent on the link_type
        p_var = self.theta_p + (self.weights_p @ action)

        next_state_prob_value = p_var  # intermediate function that goes into link function
        
        next_state = np.zeros(self.n_arms, dtype=np.float32)
        expected_reward = np.zeros(self.n_arms)
        next_state_prob = np.zeros(self.n_arms)
        assert np.all(state >= 0)
        assert np.all(self.theta_p >= 0)
        assert np.all(self.weights_p >= 0)
        for j in range(self.n_arms):
            next_state_prob[j] = self.link_function(j, next_state_prob_value[j])
            if next_state_prob[j] < 0:
                print('  prob', next_state_prob[j], next_state_prob_value[j])
                import pdb; pdb.set_trace()
            arm_state = state[j].astype(int)
            up_prob = next_state_prob[j]

            # get next state
            state_move = np.random.binomial(1, up_prob)
            if state_move == 1:
                next_state[j] = arm_state + 1
            else:
                next_state[j] = arm_state - 1

            # calculate expected reward of action
            up_reward = self.rmab.state_r[j, min(arm_state + 1, self.rmab.n_states-1)]
            down_reward = self.rmab.state_r[j, max(arm_state - 1, 0)]
            expected_reward[j] = up_reward * up_prob + down_reward * (1 - up_prob)

        # ensure we don't go above or below cap
        next_state[next_state >= self.rmab.n_states] = self.rmab.n_states - 1
        next_state[next_state < 0] = 0

        if advance:
            self.advance()
            self.set_state(next_state)
        
        rewards = np.zeros(self.n_arms, dtype=np.float32)
        for j in range(self.n_arms):
            rewards[j] = self.rmab.state_r[j, next_state[j].astype(int)]
        
        terminated = self.is_done()
        truncated = False
        info = {'next_state_expected': expected_reward.sum()}
        return next_state, rewards.sum(), terminated, truncated, info


    def link_function(self, arm, value):
        """ implement different link functions """
        if self.link_type == 'linear':
            return value
        
        elif self.link_type == 'submodular':
            return 1 - (1/(1+value))

        elif self.link_type == 'sigmoid':
            scale_down = 0.98      # scale upper limit to 0.98
            epsilon = 0.01         # add to the end to ensure we stay within 0 and 1
            if isinstance(self, MultiActionRMAB):
                y_intercept = self.a_vals[arm] * (1 / (1 + np.exp(self.x0_vals[arm])))
                probability = self.a_vals[arm] * (1 / (1 + np.exp(-(self.b_vals[arm] * value - self.x0_vals[arm])))) - y_intercept
                
            elif isinstance(self, TorchMultiActionRMAB):
                y_intercept = self.a_vals[arm] * (1 / (1 + torch.exp(self.x0_vals[arm])))
                probability = self.a_vals[arm] * (1 / (1 + torch.exp(-(self.b_vals[arm] * value - self.x0_vals[arm])))) - y_intercept
            
            return probability * scale_down + epsilon
        
        else:
            raise Exception(f'link {self.link_type} not implemented')

    
    def calc_action_expected_value(self, action):
        """ calculate expected value of an action """

        if isinstance(action, list):
            use_torch = False
            action = np.array(action)
        elif isinstance(action, np.ndarray):
            use_torch = False
        elif torch.is_tensor(action):
            use_torch = True
        else:
            raise Exception(f'action of type {type(action)} not defined')

        if use_torch:
            expected_reward = torch.zeros(self.n_arms)
        else:
            expected_reward = np.zeros(self.n_arms, dtype=np.float32)
            
        if isinstance(self.rmab, MultiStateRMAB):
            state_r = self.rmab.get_state_r()
            state = self.observation()
            if use_torch: state = state.int()
            else: state = state.astype(int)

            for j in range(self.n_arms):
                value = self.theta_p[j] + action.dot(self.weights_p[j])

                s = state[j]
                expected_prob = self.link_function(j, value)
                up_reward = state_r[j, min(s + 1, self.rmab.n_states-1)]
                down_reward = state_r[j, max(s - 1, 0)]
                expected_reward[j] = up_reward * expected_prob + down_reward * (1 - expected_prob)
        else:
            raise NotImplementedError

        return expected_reward.sum()
    
    
    def to_networkx(self):
        node_actions = [f'A{i}' for i in range(self.action_dim)]
        node_arms    = [f'X{j}' for j in range(self.n_arms)]

        G = nx.Graph()
        G.add_nodes_from(node_actions, bipartite=0)  # node attribute 'bipartite'
        G.add_nodes_from(node_arms, bipartite=1)
        for i in range(self.action_dim):
            for j in range(self.n_arms): 
                if self.weights_p[j][i] != 0:
                    G.add_edge(f'A{i}', f'X{j}', weight=self.weights_p[j][i])

        # draw graph
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

        # bipartite positioning
        top = node_actions
        pos = nx.bipartite_layout(G, top)

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=700)

        # edges
        nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
        nx.draw_networkx_edges(
            G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
        )

        # node labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis('off')
        plt.tight_layout()
        # plt.show()

        return G
    
    def get_approximator(self):
        return MultiActionRmabApproximator


class TorchMultiActionRMAB(MultiActionRMAB):
    def __init__(self, base_rmab, multi_action_rmab):

        super().__init__(base_rmab, multi_action_rmab.action_dim, init_state=multi_action_rmab.init_state, link_type=multi_action_rmab.link_type, make_weights=False)

        self.theta_p   = torch.tensor(multi_action_rmab.theta_p, dtype=torch.float32, requires_grad=False)
        self.weights_p = torch.tensor(multi_action_rmab.weights_p, dtype=torch.float32, requires_grad=False)

        self.init_state = torch.tensor(multi_action_rmab.init_state, dtype=torch.float32, requires_grad=False)
        self.state = None

        # -------------------------------------------------------
        # create torch_geometric graph
        # -------------------------------------------------------
        self.data = make_geometric(self.weights_p, base_rmab.init_state)

        self.n_nodes = self.data.num_nodes
        self.n_edges = self.data.num_edges
        self.n_action_features = self.data.num_action_features
        self.n_arm_features = self.data.num_arm_features
        self.n_node_features = self.data.num_node_features

        print('auto generated:')
        print('edge index', self.data.edge_index)
        print('edge attributes', self.data.edge_attr)
        print('node features', self.data.x)
        print('\n\n')

        # available attributes
        print(f'graph with {self.n_nodes} nodes and {self.n_edges} edges')
        print(f'  node features: {self.n_node_features}')
        print(f'  average node degree: {self.n_edges / self.n_nodes:.2f}')
        print(f'  contains isolated nodes: {self.data.has_isolated_nodes()}')
        print(f'  contains self-loops:     {self.data.has_self_loops()}')

    def __str__(self):
        return f'TorchMultiActionRMAB_self.{self.link_type}_N{self.action_dim}_J{self.n_arms}_B{self.budget}_H{self.horizon}'
    
    def get_x(self):
        raise Exception('get_x not available for BipartiteData graph')
    
    def get_action(self):
        return self.data.x_action
    
    def get_random_action(self):
        random_action = super().get_random_action()
        return torch.tensor(random_action, requires_grad=False).float()
    
    def get_null_action(self):
        return torch.zeros(self.action_dim)
    
    def get_arm(self):
        return self.data.x_arm
    
    def get_edge_index(self):
        return self.data.edge_index

    def set_action(self, action):
        assert len(action) == self.action_dim, f'action is {len(action)} long, should be {self.action_dim}'
        self.data.x[:self.action_dim, 1] = action
        
        
    def step(self, action, advance=True, allow_excess=False):
        """ allow for multiple actions to be taken simultaneously
        redo the act of MultiActionRMAB but with torch operations """
        assert len(action) == self.action_dim

        # check budget constraint
        if not allow_excess: assert action.sum() <= self.budget + 1e-5, f'action is {action} which is more than budget {self.budget}'
        
        state = self.observation()
        
        # get a transition probability based on the implemented action
        # dependent on the link_type
        p_var = self.theta_p + (self.weights_p @ action)

        next_state_prob_value = p_var
        next_state_prob = torch.zeros(self.n_arms)
        for j in range(self.n_arms):
            next_state_prob[j] = self.link_function(j, next_state_prob_value[j])
        
        next_state = torch.zeros(self.n_arms)
        expected_reward = torch.zeros(self.n_arms)
        for j in range(self.n_arms):
            arm_state = state[j].int()
            up_prob = next_state_prob[j]

            if torch.any(next_state_prob > 1) or torch.any(next_state_prob < 0):
                breakpoint()
            
            # get next state
            state_move = torch.bernoulli(up_prob)
            if state_move == 1:
                next_state[j] = arm_state + 1
            else:
                next_state[j] = arm_state - 1

            # calculate expected reward of action
            up_reward = self.rmab.state_r[j, min(arm_state + 1, self.rmab.n_states-1)]
            down_reward = self.rmab.state_r[j, max(arm_state - 1, 0)]
            expected_reward[j] = up_reward * up_prob + down_reward * (1 - up_prob)

        # ensure we don't go above or below cap
        next_state[next_state >= self.rmab.n_states] = self.rmab.n_states - 1
        next_state[next_state < 0] = 0
        
        if advance:
            self.advance()
            self.set_state(next_state)
        
        rewards = torch.zeros(self.rmab.n_arms, dtype=torch.float32)
        for j in range(self.rmab.n_arms):
            rewards[j] = self.rmab.state_r[j, next_state[j].int()]

        terminated = self.is_done()
        truncated = False
        info = {'next_state_expected': expected_reward.sum()}

        return next_state, rewards.sum(), terminated, truncated, info
