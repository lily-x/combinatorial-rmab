""" use DQN to estimate the Q-function 
coupled with MIP solver to pick action at each timestep 

DQN code adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import math
import random
import tqdm
import datetime
import warnings
from collections import namedtuple, deque

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from environment.base_envs import TorchStandardRMAB, TorchMultiStateRMAB
from environment.multi_action import TorchMultiActionRMAB
from environment.scheduling import TorchSchedulingRMAB
from environment.constrained import TorchConstrainedRMAB
from environment.routing import TorchRoutingRMAB
from model2mip.net2mip import Net2MIPPerScenario

from algos.value_estimator import get_test_trajectories
# from dqn.replay_buffer import PrioritizedReplayBuffer
# from dqn.prioritized_memory import Memory
from torchrl.data import ListStorage, PrioritizedReplayBuffer


device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##############################################################
# DQN utils
##############################################################

class Memoizer:
    """ create a memoizer that can store past solves, but will periodically clear old data """
    def __init__(self, refresh=5):
        self.refresh = refresh  # how many episodes of solves to save
        # existing_solves will be a list of dicts, where most recent items are at the front
        self.existing_solves = [{}]
        self.episode = 0

        # store the fraction of checks that actually had a success
        self.num_checks = [0]
        self.num_successes = [0]

    def reset(self):
        self.existing_solves = [{}]
        self.episode = 0

        self.num_checks = [0]
        self.num_successes = [0]

    def add(self, key, value):
        self.existing_solves[0][key] = value

    def new_episode(self):
        """ delete the oldest dict and add in a dict for new episode """
        tot_solves = np.array(self.num_checks).sum()
        tot_successes = np.array(self.num_successes).sum()
        frac = tot_successes / tot_solves
        # print(f' memoizer: episode {self.episode}, curr frac success {frac:.2f} ({tot_successes} / {tot_solves})')

        del self.existing_solves[-1]
        self.existing_solves.insert(0, {})
        self.episode += 1

        del self.num_checks[-1]
        del self.num_successes[-1]
        self.num_checks.insert(0, 0)
        self.num_successes.insert(0, 0)

        return self.episode

    def check_key(self, key):
        """ if key is already stored, return value """
        self.num_checks[0] += 1

        # iterate through solve list in terms of most recent
        for solve_list in self.existing_solves:
            if key in solve_list:
                self.num_successes[0] += 1
                return solve_list[key]
            
        return None

        
##############################################################
# old replay buffer ... before PER
##############################################################

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'cost'))  # use 'cost' not 'reward' because MIP is a minimizer


class ReplayMemory(object):
    """
    replay buffer, for experience replay:
    - stores the transitions that the agent observes, allowing us to reuse this data later
    - by sampling from it randomly, the transitions that build up a batch are decorrelated
    - has been shown to greatly stabilize and improve the DQN training procedure
    """ 
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, cost):
        """ save a transition """
        self.memory.append(Transition(state.unsqueeze(0), action.unsqueeze(0), next_state.detach().unsqueeze(0), cost.unsqueeze(0)))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def get_idx(self, idx):
        return self.memory[idx]

    def __len__(self):
        return len(self.memory)


##############################################################
# DQN network
##############################################################

class DQN(nn.Module):
    """ Q-function estimator
    possibly adapt code from: https://github.com/khalil-research/Neur2SP/blob/main/nsp/models/network.py """

    def __init__(self, in_dim, out_dim):
        self.hidden_dim = [16, 32] # [28, 28]  # 128, 128

        super(DQN, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_dim, self.hidden_dim[0])
        self.fc2 = nn.Linear(self.hidden_dim[0], self.hidden_dim[1])
        self.fc3 = nn.Linear(self.hidden_dim[1], out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


##############################################################
# DQN solver
##############################################################

class DQNSolver:
    def __init__(self, t_rmab):
        assert isinstance(t_rmab, TorchMultiActionRMAB) or isinstance(t_rmab, TorchStandardRMAB) or \
               isinstance(t_rmab, TorchMultiStateRMAB) or isinstance(t_rmab, TorchSchedulingRMAB) or \
               isinstance(t_rmab, TorchConstrainedRMAB) or isinstance(t_rmab, TorchRoutingRMAB)

        self.GAMMA = 0.99        # discount factor

        # epsilon greedy parameters
        self.EPS_START = 0.9     # starting epsilon (random prob)
        self.EPS_END = 0.05      # final epsilon
        self.EPS_DECAY = 1000    # rate of exponential decay of epsilon (higher -> slower decay)

        # optimizer parameters
        self.LR = 2.5e-4         # learning rate of optimizer
        self.ADAM_EPS = 1.5e-4
        self.GRADIENT_CLIP = 5.  # 100
        self.SCHEDULER_GAMMA = 0.9      # optimizer LR, exponential decay

        self.TARGET_UPDATE_FREQ = 300

        self.TAU = 1e-6                 # update rate of target NN

        # replay buffer parameters
        self.BATCH_SIZE = 32     # num transitions to sample from replay buffer
        self.MEMORY_SIZE = 1e4   # length of memory

        # prioritized experience replay parameters
        self.ALPHA = 0.6         # how much prioritization is used (for experience replay)
        self.BETA = 0.4          # how much importance sampling is used
        self.PRIOR_EPS = 1e-6    # guarantees every transition can be sampled

        self.MEMOIZER_REFRESH = 5 # how often (# episodes) we clear out memoizer

        # for early myopic training
        self.N_EPOCHS_MYOPIC = 300

        if torch.cuda.is_available():
            self.N_EPISODES = 100 #600
        else:
            self.N_EPISODES = 100

        self.t_rmab = t_rmab
        self.action_dim = self.t_rmab.action_dim
        self.n_arms = self.t_rmab.n_arms
        self.state = self.t_rmab.observation()
        self.budget = self.t_rmab.budget

        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.mipper = None
        self.memory = None
        self.memoizer = None


    def select_action(self, state):
        eps_sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.step_count / self.EPS_DECAY)
        self.step_count += 1

        # epsilon-greedy policy
        if eps_sample < eps_threshold:
            random_action = self.t_rmab.get_random_action()
            return random_action
        else:
            with torch.no_grad():
                # pick action based on previous expected value estimate
                results = self.approximator.approximate(self.policy_net, self.mipper, scenario_embedding=state)
                action  = results['sol']
                if not torch.is_tensor(action):
                    action = torch.tensor(action, dtype=torch.float32)  # convert to tensor for compatability with TRmab
                return action
                

    def pretrain_model_myopic(self, memory):
        """ pre-train model by using myopic expected cost
        iterate through entire training dataset """

        print('----------- pretrain model - w/ myopic costs')

        myopic_model = DQN(self.action_dim + self.n_arms, 1).to(device)
        myopic_optimizer = optim.AdamW(myopic_model.parameters(), lr=self.LR, eps=self.ADAM_EPS, amsgrad=True)
        scheduler = ExponentialLR(myopic_optimizer, gamma=self.SCHEDULER_GAMMA)
        criterion = nn.MSELoss()

        n_minibatches = math.floor(len(memory) / self.BATCH_SIZE)

        test_loss = []
        test_input, test_cost = get_test_trajectories(self.t_rmab, n_test_ep=50)

        for epoch in tqdm.tqdm(range(self.N_EPOCHS_MYOPIC)):
            for mb in range(n_minibatches):
                batch, batch_info = memory.sample(return_info=True)

                state_batch, action_batch, next_states, cost_batch = batch
                cost_batch = cost_batch.unsqueeze(1)

                nn_input = torch.concatenate((action_batch, state_batch), axis=1).requires_grad_(True)

                # in your training loop
                myopic_optimizer.zero_grad()   # zero the gradient buffers
                nn_output = myopic_model(nn_input)
                loss = criterion(nn_output, cost_batch)
                loss.mean().backward()
                myopic_optimizer.step()    # does the update

            # calculate loss on test set
            with torch.no_grad():
                test_output = myopic_model(test_input)
                loss = criterion(test_output, test_cost)
                test_loss.append(loss.mean().detach().item())

            scheduler.step()
            
        plt.figure()
        # plt.plot(np.arange(len(track_loss)), track_loss)
        plt.plot(np.arange(len(test_loss)), test_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(f'myopic training: N{self.n_arms} J{self.action_dim} {len(memory)} samples, {self.t_rmab.horizon} horizon')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        plt.savefig(f'plots/dqn_pretrain_myopic_{myopic_model.hidden_dim}_{self.t_rmab}_{timestamp}.png')
        plt.close()
        # plt.show()

        return myopic_model


    def optimize_model_run_all(self, n_epochs=5, perturb=True):
        """ iterate through the entire training dataset and optimize model
        
        NOTE: after changing to prioritized experience replay, we are no longer guaranteed to go through all samples
         (but we will iterate through the # of samples) 
         
        perturb: toggle for whether we include perturbed myopic samples """

        print('----------- optimize model run all')
        print(f'  memory len {len(self.memory)}')
        step_count = 0
        for epoch in range(n_epochs):
            n_minibatches = max(100, math.floor(len(self.memory) / self.BATCH_SIZE))

            for mb in tqdm.tqdm(range(n_minibatches)):
                batch, info = self.memory.sample(return_info=True)
                self.optimize_model(batch, batch_info=info, perturb=False)
                
                # run through the same batch twice, once with perturbations and one directly
                if perturb:
                    self.optimize_model(batch, batch_info=info, perturb=True)
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 − τ)θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)
                

                

    def optimize_model_run_single_batch(self):
        if len(self.memory) < self.BATCH_SIZE: return

        batch, info = self.memory.sample(return_info=True)

        self.optimize_model(batch, batch_info=info)


    def optimize_model(self, batch, batch_info=None, perturb=False):
        """ given a batch of samples, compute a one-step update of the model
        
        per_idx: for prioritized experience replay, keep track of indices that we'll use to update
        
        perturb: instead of using the actions directly, solve with a variant of those actions
            (only easy to implement if the only feasibility constraints are the budget feasibility) 
        """
        state_batch, action_batch, next_states, cost_batch = batch
        cost_batch = cost_batch.unsqueeze(1)

        # only perturb for cases where we only need to satisfy budget constraint
        # and all actions have the same cost
        if perturb:
            action_batch = action_batch.clone()
            if not (isinstance(self.t_rmab, TorchMultiActionRMAB) or isinstance(self.t_rmab, TorchStandardRMAB) or isinstance(self.t_rmab, TorchMultiStateRMAB)):
                warnings.warn('no perturbation for object type')
                
            # take every action and perturb 
            else:
                for k in range(self.BATCH_SIZE):
                    # if action is *not* null action, pick item to remove
                    if action_batch[k, :].sum() > 0:
                        curr_actions = torch.where(action_batch[k, :] == 1)[0]
                        rand_remove = torch.randint(0, len(curr_actions), (1,))
                        action_batch[k, rand_remove] = 0

                    # pick item to add
                    free_action = torch.where(action_batch[k, :] == 0)[0]
                    rand_add = torch.randint(0, len(free_action), (1,))
                    action_batch[k, rand_add] = 1

        # compute Q(s_t, a)
        state_action_values = self.policy_net(torch.concatenate([action_batch, state_batch], axis=1))  #.gather(1, action_batch)

        # compute V(s_{t+1}) for all next states, using "older" target_net to compute
        # expected values of actions for next_states (and MIP solver to select their best reward)
        with torch.no_grad():
            # use MIP solver to find best action
            all_best_actions = torch.zeros((self.BATCH_SIZE, self.action_dim), dtype=torch.float32)
            for k in range(self.BATCH_SIZE):
                next_state_tup = tuple(next_states[k].numpy())

                check_key = self.memoizer.check_key(next_state_tup)
                if check_key is not None:
                    best_actions = check_key
                else:
                    results = self.approximator.approximate(self.target_net,
                                                            self.mipper, scenario_embedding=next_states[k], 
                                                            # gap=0.10,
                                                            # time_limit=100,
                                                            threads=4,
                                                            # log_dir='gurobi_logs'
                                                            )
                    best_actions = results['sol']
                    self.memoizer.add(next_state_tup, best_actions)

                if torch.is_tensor(best_actions):
                    all_best_actions[k, :] = best_actions
                else:
                    all_best_actions[k, :] = torch.tensor(best_actions)  # convert to tensor for compatability with TRmab

            net_input = torch.concatenate((all_best_actions, next_states), axis=1)
            next_state_values = self.target_net(net_input)

        # compute expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + cost_batch

        # compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values)

        # optimize the model
        self.optimizer.zero_grad()
        per_weights = torch.tensor(batch_info['_weight'])
        per_idx = batch_info['index']
        (per_weights * loss).mean().backward()
        
        # in-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.GRADIENT_CLIP)
        self.optimizer.step()

        # print(f'   optimizer loss {loss.detach().item():.2f}')
        self.optimizer_loss.append(loss.mean().detach().item())

        return self.policy_net


    def get_training_data(self):
        print('----------- get training data')

        training_memory = PrioritizedReplayBuffer(alpha=self.ALPHA, beta=self.BETA, 
                          batch_size=self.BATCH_SIZE, storage=ListStorage(max_size=1e5)) 
        
        self.t_rmab.reset()

        # different actions we want to try for all states
        try_action_starter = []
        try_action_starter.append(torch.ones(self.t_rmab.action_dim))  # all 1s
        try_action_starter.append(torch.zeros(self.t_rmab.action_dim)) # all 0s

        for i in range(self.t_rmab.action_dim):                     # all 0s but one 1
            action0s = torch.zeros(self.t_rmab.action_dim, requires_grad=False)
            action1s = torch.ones(self.t_rmab.action_dim, requires_grad=False)
            action0s[i] = 1
            action1s[i] = 0
            try_action_starter.append(action0s)
            try_action_starter.append(action1s)


        # ------------------------------------------------------------
        # add strategic states
        # ------------------------------------------------------------
        try_states = []
        try_states.append(torch.ones(self.t_rmab.n_arms))  # all 1s
        try_states.append(torch.zeros(self.t_rmab.n_arms)) # all 0s
        for j in range(self.t_rmab.n_arms):                     # all 0s but one 1
            state0s = torch.zeros(self.t_rmab.n_arms, requires_grad=False)
            state1s = torch.ones(self.t_rmab.n_arms, requires_grad=False)
            state0s[j] = 1
            state1s[j] = 0
            try_states.append(state0s)
            try_states.append(state1s)

        # add a number of different actions
        # many random actions
        try_actions = [self.t_rmab.get_random_action().float() for _ in range(10)]

        for try_state in try_states:
            for try_action in try_actions + try_action_starter:
                self.t_rmab.state = try_state
                next_state, reward, terminated, truncated, info = self.t_rmab.step(try_action, advance=False, allow_excess=True)
                expected_cost = -info['next_state_expected']

                # store transition in memory
                new_transition = Transition(self.t_rmab.observation(), try_action, next_state.detach(), expected_cost)
                training_memory.add(new_transition)

        # artificially force states to be in high-potential action
        # generate a number of random states (and try several next actions for that state)
        #n_random = 100 * self.t_rmab.horizon
        n_random = 50 * self.t_rmab.horizon
        for _ in tqdm.tqdm(range(n_random)):
            if isinstance(self.t_rmab, TorchSchedulingRMAB) or isinstance(self.t_rmab, TorchConstrainedRMAB) or isinstance(self.t_rmab, TorchRoutingRMAB):
                base_rmab = self.t_rmab.rmab
            else:
                base_rmab = self.t_rmab
        
            try_actions = []
            if isinstance(base_rmab, TorchMultiStateRMAB) or isinstance(base_rmab, TorchRoutingRMAB):
                selected_arms = np.random.choice(base_rmab.n_arms, base_rmab.budget, replace=False)
                random_state = np.random.choice(base_rmab.n_states, base_rmab.n_arms, replace=True)
                random_state[selected_arms] = base_rmab.n_states - 1
                random_state = torch.tensor(random_state, dtype=torch.float32, requires_grad=False)

                # teach the value of acting vs. not acting on these arms
                action = torch.zeros(base_rmab.n_arms)
                action[selected_arms] = 1
                try_actions.append(action)

            elif isinstance(base_rmab, TorchMultiActionRMAB):
                selected_arms = np.random.choice(base_rmab.n_arms, base_rmab.budget, replace=False)
                random_state = torch.zeros(base_rmab.n_arms, requires_grad=False)
                random_state[selected_arms] = 1

                # teach the value of acting vs. not acting on these arms
                selected_actions = np.random.choice(base_rmab.action_dim, base_rmab.budget, replace=False)
                action = torch.zeros(base_rmab.action_dim, requires_grad=False)
                action[selected_actions] = 1
                try_actions.append(action)

            elif isinstance(base_rmab, TorchStandardRMAB):
                selected_arms = np.random.choice(base_rmab.n_arms, base_rmab.budget, replace=False)
                random_state = torch.zeros(base_rmab.n_arms, requires_grad=False)
                random_state[selected_arms] = 1

                # teach the value of acting vs. not acting on these arms
                action = torch.zeros(base_rmab.n_arms)
                action[selected_arms] = 1
                try_actions.append(action)


            else:
                raise NotImplementedError

            # add a number of different actions
            # many random actions
            for _ in range(10):
                try_actions.append(self.t_rmab.get_random_action().float())


            for try_action in try_actions + try_action_starter:
                self.t_rmab.reset_to_state(random_state)
                next_state, reward, terminated, truncated, info = self.t_rmab.step(try_action, advance=False, allow_excess=True)
            
                expected_cost = -info['next_state_expected']

                # store transition in memory
                new_transition = Transition(self.t_rmab.observation(), try_action, next_state.detach(), expected_cost)
                training_memory.add(new_transition)

        print(f'   {len(training_memory)} training points')

        return training_memory
                

    def training_loop(self):
        """ one iteration of the training loop """

        print('----------- begin main loop')
        print(f'  memory len {len(self.memory)}')
        for i_episode in tqdm.tqdm(range(self.N_EPISODES)):
            # initialize environment and get state
            state = self.t_rmab.reset()  # , info
            for t in range(self.t_rmab.horizon):
                state = self.t_rmab.observation()
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.t_rmab.step(action)

                expected_cost = -info['next_state_expected']

                # store transition in memory
                new_transition = Transition(state, action, next_state.detach(), expected_cost)
                self.memory.add(new_transition)

                # perform one step of the optimization (on the policy network)
                self.optimize_model_run_single_batch()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 − τ)θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                # # periodically update the target network with the policy Q network
                # if self.step_count % self.TARGET_UPDATE_FREQ == 0:
                #     self.target_net.load_state_dict(self.policy_net.state_dict())

            self.memoizer.new_episode()

            self.scheduler.step() # learning rate scheduler

        plt.figure()
        plt.plot(np.arange(len(self.optimizer_loss)), np.array(self.optimizer_loss))
        plt.title(f'DQN loss N={self.action_dim} J={self.n_arms} budget={self.budget} horizon={self.t_rmab.horizon}')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        plt.savefig(f'plots/dqn_loss_{self.policy_net.hidden_dim}_{self.t_rmab}_{timestamp}.png')
        # plt.show()


    def train(self, fast_prototyping=False):
        """ 
        fast_prototyping - an optional parameter to skip the training loop for fast prototyping"""

        print('----------- training DQN: myopic supervised learning')

        # initialize training components
        
        # for tracking stats
        self.optimizer_loss = []
        self.step_count = 0
        
        # pre-training process
        training_memory = self.get_training_data()
        myopic_model = self.pretrain_model_myopic(training_memory)
        # plt.show()
        # initialize sequential training components
        self.policy_net = DQN(self.action_dim + self.n_arms, 1).to(device)
        self.policy_net.load_state_dict(myopic_model.state_dict())

        self.target_net = DQN(self.action_dim + self.n_arms, 1).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, eps=self.ADAM_EPS, amsgrad=True)
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        self.memoizer = Memoizer(refresh=self.MEMOIZER_REFRESH)
        self.mipper = Net2MIPPerScenario
        self.approximator = self.t_rmab.get_approximator()(self.t_rmab)

        midway_net = DQN(self.action_dim + self.n_arms, 1).to(device)
        midway_net.load_state_dict(self.policy_net.state_dict())

        # move training samples to smaller storage
        # only fill 10% of storage to avoid flooding with bad samples
        training_memory_samples = random.sample(training_memory._storage._storage, 
                                                int(min(self.MEMORY_SIZE / 10, len(training_memory))))
        self.memory = PrioritizedReplayBuffer(alpha=self.ALPHA, beta=self.BETA, eps=self.PRIOR_EPS, 
                          batch_size=self.BATCH_SIZE, storage=ListStorage(max_size=self.MEMORY_SIZE))
        self.memory.extend(training_memory_samples)

        if not fast_prototyping:
            # run real DQN training loop
            self.training_loop()

        return self.policy_net, myopic_model, midway_net


    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
