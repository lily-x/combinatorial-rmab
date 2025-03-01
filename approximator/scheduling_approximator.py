import time

import gurobipy as gp
from gurobipy import GRB

from approximator.standard_rmab_approximator import StandardRmabApproximator
from environment.base_envs import MultiStateRMAB
from approximator.multistate_approximator import optimize_multistate_myopic

class SchedulingRmabApproximator(StandardRmabApproximator):
    def __init__(self, rmab, model_type='NN-E'):
        """ rmab: SchedulingRmab """
        super().__init__(rmab, model_type)
        self.action_dim = rmab.n_arms


    def get_master_mip(self):
        """
        initialize MIP model with first-stage variables and constraints
        
        note: objective needs to be added elsewhere """
        n_arms = self.rmab.n_arms
        budget = self.rmab.budget
        n_workers = self.rmab.n_workers
        n_timeslots = self.rmab.n_timeslots

        # set up Gurobi optimizer --------------------------------------------------
        model = gp.Model('rmab_scheduling')
        model.setParam('OutputFlag', 0) # silence output
        # model.setParam('IterationLimit', max_iterations) # limit number of simplex iterations


        # define variables ---------------------------------------------------------
        x_vars = [[[] for j in range(n_arms)] for i in range(n_workers)]
        for i in range(n_workers):
            for j in range(n_arms):
                for s in range(n_timeslots):
                    if self.rmab.arm_M[j][s] == self.rmab.worker_M[i][s] == 1:
                        x_vars[i][j].append(model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}_{s}'))
                    else:
                        x_vars[i][j].append(0)

        # define pulling actions
        to_pull = [model.addVar(vtype=GRB.BINARY, name=f'action_{j}')
                    for j in range(n_arms)]

        # define constraints -------------------------------------------------------
        # total budget constraint
        model.addConstr((gp.quicksum(to_pull) <= budget), 'budget')
        
        # each worker gets assigned only once per timeslot
        model.addConstrs(((gp.quicksum([x_vars[i][j][s] for j in range(n_arms)]) <= 1) 
                          for i in range(n_workers) for s in range(n_timeslots)), 'worker_only_once_per_timeslot')
        
        # each arm gets assigned twice (if available)
        model.addConstrs(((gp.quicksum([x_vars[i][j][s] for i in range(n_workers) for s in range(n_timeslots)]) <= 2) 
                          for j in range(n_arms)), 'arm_only_once')
        
        # each assigned arm has two workers
        # p_j = max {x_ijs}
        model.addConstrs(((2 * to_pull[j] <= gp.quicksum([x_vars[i][j][s] for i in range(n_workers) for s in range(n_timeslots)])) 
                          for j in range(n_arms)), 'pulling_action_bottom')
        
        model.addConstrs(((to_pull[j] >= x_vars[i][j][s]) 
                          for i in range(n_workers) for j in range(n_arms) for s in range(n_timeslots)), 'pulling_action_upper')
        
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
            return super().optimize()
    
        