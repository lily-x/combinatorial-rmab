""" use a NN to estimate the value function (one-step) """

import math, random
import datetime
import numpy as np
from sklearn import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import tqdm
import matplotlib.pyplot as plt

from environment.base_envs import TorchStandardRMAB, TorchMultiStateRMAB
from environment.multi_action import TorchMultiActionRMAB

from model2mip.net2mip import Net2MIPPerScenario
from models.network import ReLUNetworkPerScenario


##############################################################
# generate test data
##############################################################

def get_test_trajectories(t_rmab, n_test_ep=10):
    """ generate random trajectores for test dataset """
    test_input = []
    test_cost = []
    for episode in range(n_test_ep):
        t_rmab.fresh_reset()

        for t in range(t_rmab.horizon):
            # sample = random.random()

            action = t_rmab.get_random_action()
            
            # calculate expected value for action
            # NOTE: take negative because uses costs
            expected_cost = -t_rmab.calc_action_expected_value(action)

            test_input.append(torch.concatenate([action, t_rmab.observation()]).detach().numpy())
            test_cost.append(expected_cost.unsqueeze(0).detach().numpy())

            # advance
            t_rmab.step(action)

    test_input = torch.tensor(np.array(test_input)).float()
    test_cost = torch.tensor(np.array(test_cost)).float()

    return test_input, test_cost


##############################################################
# dataset class
##############################################################

class RmabDataset(Dataset):
    """dataset for RMAB myopic reward functions"""

    def __init__(self, inputs, rewards):
        """ 
        inputs - [state, action]
        rewards - value
        """
        self.inputs = inputs
        self.rewards = rewards

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.inputs[idx]
        rewards = self.rewards[idx]
        sample = {'input': input, 'reward': rewards}

        return sample


##############################################################
# training process to estimate value function
##############################################################

def estimate_val_func(t_rmab):
    """ use sampled (action, reward) pairs to learn value function """

    assert isinstance(t_rmab, TorchMultiActionRMAB) or isinstance(t_rmab, TorchStandardRMAB) or isinstance(t_rmab, TorchMultiStateRMAB)
    
    n_arms = t_rmab.n_arms

    if isinstance(t_rmab, TorchMultiActionRMAB):
        action_dim = t_rmab.action_dim
        in_dim  = action_dim + n_arms  # size of state and actions
    elif isinstance(t_rmab, TorchStandardRMAB):
        in_dim = 2 * n_arms
    elif isinstance(t_rmab, TorchMultiStateRMAB):
        in_dim = 2 * n_arms

    net = ReLUNetworkPerScenario(in_dim, [20, 20])

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.MSELoss()

    # set up MIP estimator for this round
    mipper = Net2MIPPerScenario
    approximator = t_rmab.get_approximator()(t_rmab)

    n_episodes = 50

    track_loss = []

    test_input, test_cost = get_test_trajectories(t_rmab)
    test_loss = []

    all_input = []
    all_cost = []

    eps_threshold = 0.1

    for episode in tqdm.tqdm(range(n_episodes)):
        t_rmab.fresh_reset()

        for iter in range(t_rmab.horizon):
            sample = random.random()

            # begin with 10 episodes of random actions
            if episode < 10 or sample < eps_threshold:
                action = t_rmab.get_random_action()
            else:
                # pick action based on previous expected value estimate
                results = approximator.approximate(net, mipper, scenario_embedding = t_rmab.observation())
                action  = results['sol']
                action  = torch.tensor(action, dtype=torch.float32)  # convert to tensor for compatability with TRmab

            # calculate expected value for action
            # NOTE: take negative because uses costs
            expected_cost = -t_rmab.calc_action_expected_value(action)

            all_input.append(torch.concatenate([action, t_rmab.observation()]).detach().numpy())
            all_cost.append(expected_cost.unsqueeze(0).detach().numpy())

            # advance
            t_rmab.step(action)

        # import pdb; pdb.set_trace()
        n_epochs = 10
        batch_size = 32
        for epoch in range(n_epochs):
            all_input_torch = torch.tensor(all_input)
            all_cost_torch = torch.tensor(all_cost)
            # shuffle data
            all_input_shuf, all_cost_shuf = utils.shuffle(all_input_torch, all_cost_torch) 
            
            for batch in range(math.floor(len(all_input) / batch_size)):
                # get minibatch for training
                nn_input = all_input_shuf[batch*batch_size : (batch+1) * batch_size, :]
                costs = all_cost_shuf[batch*batch_size : (batch+1) * batch_size]

                # in your training loop
                optimizer.zero_grad()   # zero the gradient buffers
                nn_output = net(nn_input)
                loss = criterion(nn_output, costs)
                loss.backward()
                optimizer.step()    # does the update

                track_loss.append(loss.detach())

            # calculate loss on test set
            with torch.no_grad():
                test_output = net(test_input)
                loss = criterion(test_output, test_cost)
                test_loss.append(loss.detach())

        
    plt.figure()
    # plt.plot(np.arange(len(track_loss)), track_loss)
    plt.plot(np.arange(len(test_loss)), test_loss)
    plt.xlabel('timestep')
    plt.ylabel('loss')
    plt.title(f'value function training: {n_episodes} episodes, {t_rmab.horizon} horizon')
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig(f'plots/value_func_{timestamp}.png')
    # plt.show()
    plt.close()

    return net

