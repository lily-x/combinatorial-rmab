import time

import gurobipy as gp
from gurobipy import GRB

from approximator.standard_rmab_approximator import StandardRmabApproximator
from environment.base_envs import MultiStateRMAB
from approximator.multistate_approximator import optimize_multistate_myopic


class ConstrainedRmabApproximator(StandardRmabApproximator):
    def __init__(self, rmab, model_type='NN-E'):
        """ rmab: ConstrainedRmab """
        super().__init__(rmab, model_type)
        self.action_dim = rmab.n_arms


    def get_master_mip(self):
        """
        initialize MIP model with first-stage variables and constraints
        
        note: objective needs to be added elsewhere """
        n_arms = self.rmab.n_arms
        n_workers = self.rmab.n_workers
        arm_costs = self.rmab.arm_costs
        capacity = self.rmab.worker_capacity

        # set up Gurobi optimizer --------------------------------------------------
        model = gp.Model('rmab_constrained')
        model.setParam('OutputFlag', 0) # silence output
        # model.setParam('IterationLimit', max_iterations) # limit number of simplex iterations

        # define variables ---------------------------------------------------------
        x_vars = [[model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
                   for j in range(n_arms)] for i in range(n_workers)]
        
        x_vars = []
        for i in range(n_workers):
            x_vars.append([])
            for j in range(n_arms):
                x_vars[i].append(model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}'))

        # define pulling actions
        to_pull = [model.addVar(vtype=GRB.BINARY, name=f'action_{j}')
                    for j in range(n_arms)]

        # define constraints -------------------------------------------------------
        # per-worker budget is satisfied
        model.addConstrs(((gp.quicksum([arm_costs[j] * x_vars[i][j] for j in range(n_arms)]) <= capacity[i])
                          for i in range(n_workers)), 'per-worker_budget')
        
        # p_j = max {x_ij}
        model.addConstrs(((to_pull[j] <= gp.quicksum([x_vars[i][j] for i in range(n_workers)])) 
                          for j in range(n_arms)), 'pulling_action_bottom')
        
        model.addConstrs(((to_pull[j] >= x_vars[i][j]) 
                          for i in range(n_workers) for j in range(n_arms)), 'pulling_action_upper')
        
        model.update()
        return model
    

    def optimize_myopic(self):
        # look at sub RMAB
        if isinstance(self.rmab.rmab, MultiStateRMAB):
            total_time = time.time()
            model = self.get_master_mip()
            actions = self.get_first_stage_variables(model)
            optimize_multistate_myopic(self.rmab.rmab, model, actions)
    
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
                'solving_time': model.runtime
            }
            return results
        else:
            print('OPTIMIZE STANDARD')
            return super().optimize()
    