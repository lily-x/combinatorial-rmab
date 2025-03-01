""" generic environment classes for multi-action, combinatorial RMAB instances 
"""

import numpy as np
import gym

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T


from approximator.standard_rmab_approximator import StandardRmabApproximator
from approximator.multistate_approximator import MultiStateRmabApproximator

class BaseRMAB(gym.Env):
    def __init__(self, n_arms, action_dim, budget, horizon, 
                init_state=None):
        self.n_arms = n_arms
        self.action_dim = action_dim
        self.budget = budget
        self.horizon = horizon

        if init_state is None:
            init_state = np.random.binomial(1, 0.5, n_arms)
        self.init_state = np.array(init_state)
        self.state = init_state
        self.transitions = None

        # dynamic attributes about RMAB
        self.t     = 0

    def __str__(self):
        return 'BaseRMAB'

    def reset_to_state(self, state):
        self.reset()
        if torch.is_tensor(self.init_state) and not torch.is_tensor(state):
            self.set_state(torch.tensor(state, dtype=torch.float32, requires_grad=False))
        else:
            self.set_state(state)
        return self.state

    def set_state(self, state):
        self.state = state

    def is_done(self):
        return self.t >= self.horizon
    
    def advance(self):
        self.t += 1
    
    def observation(self):
        return self.state
    
    def get_transitions(self):
        return self.transitions

    def reset(self):
        return self.fresh_reset()

    def fresh_reset(self):
        """ new random state """
        self.t = 0
        new_state = np.random.binomial(1, 0.5, self.n_arms)
        
        if torch.is_tensor(self.init_state):
            self.set_state(torch.tensor(new_state, dtype=torch.float32, requires_grad=False))
        else:
            self.set_state(new_state)
        
        return self.observation()

    
    def step(self, action, advance=True, allow_excess=False):
        raise NotImplementedError
    
    def get_random_action(self):
        raise NotImplementedError
    
    def get_null_action(self):
        raise NotImplementedError
    
    def calc_action_expected_value(self, action):
        """ expected myopic (single-step) reward """
        raise NotImplementedError

    def get_constraints(self, mip):
        """ given a gurobi model, add action feasibility constraints """
        raise NotImplementedError

    def get_approximator(self):
        """ get MIP approximator with action constraints """
        raise NotImplementedError


##############################################################
# multi-state
##############################################################

class MultiStateRMAB(BaseRMAB):
    """ 
    RMAB with more than 2 states
    The foundation used for many of these other domains, where this 
    represnts the MDPs of each inner arm 
    """
    def __init__(self, n_arms, budget, horizon, transitions,
                 n_states=3,
                init_state=None, state_r=None):
        assert transitions.shape == (n_arms, 2, 2)

        action_dim = n_arms
        if init_state is None:
            init_state = np.random.choice(n_states, n_arms)

        super().__init__(n_arms, action_dim, budget, horizon, init_state=init_state)

        self.transitions = transitions

        if state_r is None:
            state_r_bad = np.array([0.1, 0.15, 0.2, 1]).reshape(1, -1)
            state_r_good = np.array([0.2, 0.15, 0.1, 1]).reshape(1, -1)
            state_r = state_r_bad.repeat(n_arms, axis=0)
            state_r[:budget, :] = state_r_good

            top_r_good = np.random.choice(np.arange(4, 7), budget, replace=True)
            top_r_bad = np.random.choice(np.arange(1, 4), n_arms - budget, replace=True) 

            state_r[:budget, -1] = top_r_good
            state_r[budget:, -1] = top_r_bad
            print('state rewards', state_r)
            
        self.state_r = state_r
        self.n_states = self.state_r.shape[1]

        print(f' multistate: max reward {state_r[:, -1].sum()}   {state_r[:, -1]}')


    def __str__(self):
        return f'MultiStateRMAB_J{self.n_arms}_B{self.budget}_H{self.horizon}_n_states{self.n_states}'

    def get_state_r(self):
        return self.state_r

    def step(self, action, advance=True, allow_excess=False):
        assert len(action) == self.n_arms
        if not allow_excess: assert action.sum() <= self.budget + 1e-5, f'action is {action} which is more than budget {self.budget}'

        next_state = np.zeros(self.n_arms)
        expected_reward = np.zeros(self.n_arms)
        for j in range(self.n_arms):
            state = self.state[j].astype(int)
            a = action[j].astype(int)
            up_prob = self.transitions[j, 0, a]

            # get next state
            state_move = np.random.binomial(1, up_prob)
            if state_move == 1:
                next_state[j] = state + 1
            else:
                next_state[j] = state - 1

            # calculate expected reward of action
            up_reward = self.state_r[j, min(state + 1, self.n_states-1)]
            down_reward = self.state_r[j, max(state - 1, 0)]
            expected_reward[j] = up_reward * up_prob + down_reward * (1 - up_prob)

        # ensure we don't go above or below cap
        next_state[next_state >= self.n_states] = self.n_states - 1
        next_state[next_state < 0] = 0

        if advance:
            self.advance()
            self.set_state(next_state)

        rewards = np.zeros(self.n_arms, dtype=np.float32)
        for j in range(self.n_arms):
            rewards[j] = self.state_r[j, next_state[j].astype(int)]

        terminated = self.is_done()
        truncated = False
        info = {'next_state_expected': expected_reward.sum()}
        return next_state, rewards.sum(), terminated, truncated, info
    
    def get_random_action(self):
        selected_actions = np.random.choice(self.n_arms, self.budget, replace=False)
        random_action = np.zeros(self.n_arms, dtype=np.float32)
        random_action[selected_actions] = 1

        return random_action
    
    def get_null_action(self):
        return np.zeros(self.n_arms, dtype=np.float32)
    
    def calc_action_expected_value(self, action):
        assert len(action) == self.n_arms

        expected_reward = np.zeros(self.n_arms, dtype=np.float32)
        for j in range(self.n_arms):
            s = self.state[j].astype(int)
            a = action[j].astype(int)
            
            up_prob = self.transitions[j, 0, a]
            up_reward = self.state_r[j, min(s + 1, self.n_states-1)]
            down_reward = self.state_r[j, max(s - 1, 0)]
            expected_reward[j] = up_reward * up_prob + down_reward * (1 - up_prob)

        return expected_reward.sum()

    def get_constraints(self, mip):
        """ given a gurobi model, add action feasibility constraints """
        raise NotImplementedError

    def get_approximator(self):
        """ get MIP approximator with action constraints """
        return MultiStateRmabApproximator


class TorchMultiStateRMAB(MultiStateRMAB):
    """ multi-state RMAB """
    def __init__(self, n_arms, budget, horizon, transitions,
                 n_states=4, init_state=None, state_r=None):
        super().__init__(n_arms, budget, horizon, transitions, n_states=n_states, init_state=init_state, state_r=state_r)

        self.state_r = torch.tensor(self.state_r, dtype=torch.float32, requires_grad=False)
        self.transitions = torch.tensor(transitions, dtype=torch.float32, requires_grad=False)
        self.init_state = torch.tensor(self.init_state, dtype=torch.float32, requires_grad=False)
        self.state = self.init_state
        
    def __str__(self):
        return f'TorchMultiStateRMAB_J{self.n_arms}_B{self.budget}_H{self.horizon}_n_states{self.n_states}'
    
    def step(self, action, advance=True, allow_excess=False):
        """ execute next step
        
        toggles for evaluation
        -- advance: if False, will not move to next state or advance time t 
        -- allow_excess: enables us to evaluate the reward of actions """
        assert len(action) == self.n_arms
        if not allow_excess:
            assert action.sum() <= self.budget + 1e-5, f'action is {action} which is more than budget {self.budget}'

        next_state = torch.zeros(self.n_arms)
        expected_reward = torch.zeros(self.n_arms)
        for j in range(self.n_arms):
            state = self.state[j].int()
            a = action[j].int()
            up_prob = self.transitions[j, 0, a]

            # get next state
            state_move = torch.bernoulli(up_prob)
            if state_move == 1:
                next_state[j] = state + 1
            else:
                next_state[j] = state - 1

            # calculate expected reward of action
            up_reward = self.state_r[j, min(state + 1, self.n_states-1)]
            down_reward = self.state_r[j, max(state - 1, 0)]
            expected_reward[j] = up_reward * up_prob + down_reward * (1 - up_prob)

        # ensure we don't go above or below cap
        next_state[next_state >= self.n_states] = self.n_states - 1
        next_state[next_state < 0] = 0

        if advance:
            self.advance()
            self.set_state(next_state)

        rewards = torch.zeros(self.n_arms)
        for j in range(self.n_arms):
            rewards[j] = self.state_r[j, next_state[j].int()]

        terminated = self.is_done()
        truncated = False
        info = {'next_state_expected': expected_reward.sum()}
        return next_state, rewards.sum(), terminated, truncated, info
    
    def get_random_action(self):
        random_action = super().get_random_action()
        return torch.tensor(random_action, requires_grad=False).float()
    
    def get_null_action(self):
        return torch.zeros(self.n_arms)
    
    def calc_action_expected_value(self, action):
        assert len(action) == self.n_arms

        expected_reward = torch.zeros(self.n_arms)
        for j in range(self.n_arms):
            state = self.state[j].int().item()
            a = action[j].int()
            
            up_prob = self.transitions[j, 0, a]
            up_reward = self.state_r[j, min(state + 1, self.n_states-1)]
            down_reward = self.state_r[j, max(state - 1, 0)]
            expected_reward[j] = up_reward * up_prob + down_reward * (1 - up_prob)

        return expected_reward.sum()

    def get_constraints(self, mip):
        """ given a gurobi model, add action feasibility constraints """
        raise NotImplementedError

    def get_approximator(self):
        """ get MIP approximator with action constraints """
        return MultiStateRmabApproximator



##############################################################
# standard RMAB
##############################################################

class StandardRMAB(BaseRMAB):
    """ standard binary-action, binary-state RMAB """
    def __init__(self, n_arms, budget, horizon, transitions,
                init_state=None):
        assert transitions.shape == (n_arms, 2, 2)
        action_dim = n_arms
        super().__init__(n_arms, action_dim, budget, horizon, init_state=init_state)
        self.transitions = transitions

    def __str__(self):
        return f'StandardRMAB_J{self.n_arms}_B{self.budget}_H{self.horizon}'
    
    def step(self, action, advance=True, allow_excess=False):
        assert len(action) == self.n_arms
        if not allow_excess: assert action.sum() <= self.budget + 1e-5, f'action is {action} which is more than budget {self.budget}'

        next_state = np.zeros(self.n_arms, dtype=np.float32)
        for j in range(self.n_arms):
            s = self.state[j].astype(int)
            a = action[j].astype(int)
            
            next_state_prob = self.transitions[j, s, a]
            
            next_state[j] = np.random.binomial(1, next_state_prob)

        if advance:
            self.advance()
            self.set_state(next_state)

        reward = np.sum(next_state)
        terminated = self.is_done()
        truncated = False
        info = {'next_state_expected': next_state_prob.sum()}
        return next_state, reward, terminated, truncated, info
    
    def get_random_action(self):
        selected_actions = np.random.choice(self.n_arms, self.budget, replace=False)
        random_action = np.zeros(self.n_arms, dtype=np.float32)
        random_action[selected_actions] = 1

        return random_action
    
    def get_null_action(self):
        return np.zeros(self.n_arms, dtype=np.float32)
    
    def calc_action_expected_value(self, action):
        assert len(action) == self.n_arms

        expected_reward = np.zeros(self.n_arms, dtype=np.float32)
        for j in range(self.n_arms):
            s = self.state[j].astype(int)
            a = action[j].astype(int)
            
            expected_reward[j] = self.transitions[j, s, a]

        return expected_reward.sum()

    def get_constraints(self, mip):
        """ given a gurobi model, add action feasibility constraints """
        raise NotImplementedError


    def get_approximator(self):
        """ get MIP approximator with action constraints """
        return StandardRmabApproximator


class TorchStandardRMAB(StandardRMAB):
    """ standard binary-action, binary-state RMAB """
    def __init__(self, n_arms, budget, horizon, transitions,
                init_state=None):
        super().__init__(n_arms, budget, horizon, transitions, init_state=init_state)

        self.init_state = torch.tensor(init_state, dtype=torch.float32, requires_grad=False)
        self.state = self.init_state
        
        self.transitions = torch.tensor(transitions, dtype=torch.float32, requires_grad=False)
    
    def __str__(self):
        return f'TorchStandardRMAB_J{self.n_arms}_B{self.budget}_H{self.horizon}'
    
    def step(self, action, advance=True, allow_excess=False):
        assert len(action) == self.n_arms
        if not allow_excess: assert action.sum() <= self.budget + 1e-5, f'action is {action} which is more than budget {self.budget}'

        next_state_prob = torch.zeros(self.n_arms)
        for j in range(self.n_arms):
            s = self.state[j].int()
            a = action[j].int()
            next_state_prob[j] = self.transitions[j, s, a]
        next_state = torch.bernoulli(next_state_prob)

        if advance:
            self.advance()
            self.set_state(next_state)

        reward = next_state.sum()
        terminated = self.is_done()
        truncated = False
        info = {'next_state_expected': next_state_prob.sum()}
        return next_state, reward, terminated, truncated, info
    
    
    def get_random_action(self):
        random_action = super().get_random_action()
        return torch.tensor(random_action, requires_grad=False)
    
    def get_null_action(self):
        return torch.zeros(self.n_arms)
    
    def calc_action_expected_value(self, action):
        assert len(action) == self.n_arms

        expected_reward = torch.zeros(self.n_arms)
        for j in range(self.n_arms):
            s = self.state[j].int()
            a = action[j].int()
            
            expected_reward[j] = self.transitions[j, s, a]

        return expected_reward.sum()

    def get_constraints(self, mip):
        """ given a gurobi model, add action feasibility constraints """
        raise NotImplementedError
    
