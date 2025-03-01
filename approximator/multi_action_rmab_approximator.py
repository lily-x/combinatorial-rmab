import time
import pickle

import gurobipy as gp
from gurobipy import GRB

from approximator.rmab_approximator import RmabApproximator
from environment.base_envs import MultiStateRMAB


##############################################################
# multi-action RMAB
##############################################################

class MultiActionRmabApproximator(RmabApproximator):
    def __init__(self, rmab, model_type='NN-E'):
        """ 
        rmab: MultiActionRMAB """
        super().__init__(rmab, model_type)
        self.action_dim = rmab.action_dim

        # for non-linear link functions, make a PWL approximation
        self.x_breaks = {}
        self.y_breaks = {}
        if rmab.link_type == 'sigmoid':
            a_vals = rmab.a_vals
            b_vals = rmab.b_vals
            x0_vals = rmab.x0_vals
            for j in range(rmab.n_arms):
                file_in = f'./environment/pwl/sigmoid_a{a_vals[j]:.1f}_b{b_vals[j]:.1f}_x{x0_vals[j]:.0f}_breaks10.pickle'
                with open(file_in, 'rb') as f:
                    pwl_approx = pickle.load(f)
                    self.x_breaks[j] = pwl_approx['x_breaks']
                    self.y_breaks[j] = pwl_approx['y_breaks']
                    self.y_breaks[j][0] = 0  # ensure that 0 maps to 0
                

    def get_master_mip(self):
        """
        initialize MIP model with first-stage variables and constraints """

        action_dim = self.rmab.action_dim
        n_arms    = self.rmab.n_arms
        theta_p   = self.rmab.theta_p
        weights_p = self.rmab.weights_p
        budget    = self.rmab.budget

        # set up Gurobi optimizer --------------------------------------------------
        model = gp.Model('multi_action_rmab')
        model.setParam('OutputFlag', 0) # silence output
        # model.setParam('IterationLimit', max_iterations) # limit number of simplex iterations

        if not self.rmab.link_type == 'sigmoid':
            raise NotImplementedError
        
        # define variables ---------------------------------------------------------
        # decision variables for which action to take
        actions = [model.addVar(vtype=GRB.BINARY, name=f'action_{i}')
                    for i in range(action_dim)]

        # define constraints -------------------------------------------------------
        model.addConstr((gp.quicksum(actions) == budget), 'budget')

        model.update()
        return model


    def optimize_myopic(self):
        """ run optimizer and solve 

        calculate optimal action if myopically only considering one timestep
        for linear link function only """

        n_arms     = self.rmab.n_arms
        action_dim = self.rmab.action_dim
        state      = self.rmab.observation()
        theta_p    = self.rmab.theta_p
        weights_p  = self.rmab.weights_p

        total_time = time.time()
        model = self.get_master_mip()

        actions = self.get_first_stage_variables(model)

        # set myopic objective
        self.prob_vars = [model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f'arm_prob_{j}')
                        for j in range(n_arms)]
        
        pre_pwl_prob = [model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f'pre_pwl_prob_{j}')
                        for j in range(n_arms)]

        # add PWL estimates of reward function
        for j in range(n_arms):
            model.addConstr((
                pre_pwl_prob[j] == theta_p[j] + gp.quicksum([actions[i] * weights_p[j][i] for i in range(action_dim)])), 'set_pwl')

            model.addGenConstrPWL(pre_pwl_prob[j], self.prob_vars[j], self.x_breaks[j], self.y_breaks[j], f'pwl_{j}')


        if not isinstance(self.rmab.rmab, MultiStateRMAB): raise NotImplementedError
        state_r = self.rmab.rmab.get_state_r()  # reward of each arm in each state

        expected_reward = []
        for j in range(n_arms):
            s = state[j].astype(int)

            up_reward = state_r[j, min(s + 1, self.rmab.rmab.n_states-1)]
            down_reward = state_r[j, max(s - 1, 0)]
            expected_reward.append(up_reward * self.prob_vars[j] + down_reward * (1 - self.prob_vars[j]))

        model.setObjective(-gp.quicksum(expected_reward), GRB.MINIMIZE)

        # calculate results
        model.optimize()

        try:
            obj_val = model.objVal
            first_stage_sol = self.get_first_stage_solution(model)

        except:
            print(f'model status {model.Status}')
            if model.Status == GRB.OPTIMAL:
                print('model optimal')
            elif model.Status == GRB.INFEASIBLE:
                print('infeasible')
            elif model.Status == GRB.INF_OR_UNBD:
                print('infeasible or unbounded')
            elif model.Status == GRB.UNBOUNDED:
                print('unbounded')
            elif model.Status == GRB.CUTOFF:
                print('past cutoff')
            elif model.Status in [GRB.ITERATION_LIMIT, GRB.NODE_LIMIT, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]:
                print('past iteration/node/time/solution limit')
            
            print(f'  get_first_stage_solution() with state {self.rmab.state} is unsuccessful...')
            
            # pick random action within budget
            first_stage_sol = self.rmab.get_random_action()
            obj_val = -1
            
        total_time = time.time() - total_time

        results = {
            'time': total_time,
            'predicted_obj': obj_val,
            'sol': first_stage_sol,
            # 'solving_results': solving_results,
            'solving_time': model.runtime
        }
        return results

