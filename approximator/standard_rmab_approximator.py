import time

import numpy as np
import torch

import gurobipy as gp
from gurobipy import GRB

from approximator.rmab_approximator import RmabApproximator


##############################################################
# standard RMAB
##############################################################

class StandardRmabApproximator(RmabApproximator):
    def __init__(self, rmab, model_type='NN-E'):
        """ 
        rmab: MultiActionRMAB """
        super().__init__(rmab, model_type)
        self.action_dim = rmab.n_arms


    def get_master_mip(self):
        """
        initialize MIP model with first-stage variables and constraints """
        n_arms    = self.rmab.n_arms
        state     = self.rmab.observation()
        budget    = self.rmab.budget
        transitions = self.rmab.get_transitions()

        # set up Gurobi optimizer --------------------------------------------------
        model = gp.Model('standard_rmab')
        model.setParam('OutputFlag', 0) # silence output
        # model.setParam('IterationLimit', max_iterations) # limit number of simplex iterations

        # define variables ---------------------------------------------------------
        # decision variables for which action to take
        actions = [model.addVar(vtype=GRB.BINARY, name=f'action_{j}')
                    for j in range(n_arms)]
        
        # define constraints -------------------------------------------------------
        model.addConstr((gp.quicksum(actions) <= budget), 'budget')

        model.update()
        return model
    

    def optimize_myopic(self):
        n_arms = self.rmab.n_arms
        transitions = self.rmab.get_transitions()

        total_time = time.time()
        model = self.get_master_mip()
        actions = self.get_first_stage_variables(model)

        # define objective ---------------------------------------------------------
        expected_reward = []
        for j in range(n_arms):
            state = self.rmab.observation()[j].astype(int)
            
            if state == 2:
                breakpoint()

            transition_prob = transitions[j, state, 0] * (1 - actions[j]) + transitions[j, state, 1] * actions[j]
            expected_reward.append(transition_prob)
    
        model.setObjective(-gp.quicksum(expected_reward), GRB.MINIMIZE)
    
        # calculate results
        model.optimize()
        try:
            first_stage_sol = self.get_first_stage_solution(model)
            obj_val = model.objVal

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
            
            print(f'  get_first_stage_solution() with state {self.rmab.observation()} is unsuccessful...')
            
            # pick random action within budget
            first_stage_sol = self.rmab.get_random_action()
            obj_val = -1
            
        total_time = time.time() - total_time

        results = {
            'time': total_time,
            'predicted_obj': obj_val,
            'sol': first_stage_sol,
            # 'solving_results': solving_results,
            'solving_time': model.runtime,
            'model': model
        }
        return results
    