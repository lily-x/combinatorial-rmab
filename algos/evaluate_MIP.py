"""
evaluate the DQN-MIP solver
"""
import numpy as np
from environment.routing import RoutingRMAB

def MIP_results(rmab, network, mipper, init_states):
    """ use NN embedded MIP to pick action at each timestep """
    n_episodes = len(init_states)
    approximator = rmab.get_approximator()(rmab)
    rewards = np.zeros(rmab.horizon * n_episodes)
    for ep in range(n_episodes):
        rmab.reset_to_state(init_states[ep])
        
        for t in range(rmab.horizon):
            if isinstance(rmab, RoutingRMAB):
                results = approximator.approximate(network, mipper, scenario_embedding=rmab.observation(), with_combinatorial=True)
            else:
                results = approximator.approximate(network, mipper, scenario_embedding=rmab.observation())
            
            action  = results['sol'].astype(int)
            # print(f'   MIP t {t} action {action} s {rmab.observation()}')

            _, reward, _, _, info = rmab.step(action)
            rewards[ep*rmab.horizon + t] = info['next_state_expected']

    return rewards
