import time

import numpy as np
import torch

import gurobipy as gp
from gurobipy import GRB

from approximator.standard_rmab_approximator import StandardRmabApproximator
from environment.base_envs import MultiStateRMAB
from approximator.multistate_approximator import optimize_multistate_myopic


class RoutingRmabApproximator(StandardRmabApproximator):
    def __init__(self, rmab, model_type='NN-E'):
        """ rmab: RoutingRmab """
        super().__init__(rmab, model_type)
        self.action_dim = rmab.n_arms


    def get_master_mip(self, with_combinatorial=False):
        """
        initialize MIP model with first-stage variables and constraints
        note: objective needs to be added elsewhere

        given a gurobi model, add action feasibility constraints. 
        only add the complex combinatorial constraints when with_combinatorial == True. 
        that way, we save considerable computing cost in the training process """
        n_arms = self.rmab.n_arms
        budget = self.rmab.budget
        max_path_length = self.rmab.max_path_length
        source = self.rmab.source
        edge_list = self.rmab.edge_list


        # set up Gurobi optimizer --------------------------------------------------
        model = gp.Model('rmab_routing')
        model.setParam('OutputFlag', 0) # silence output
        # model.setParam('IterationLimit', max_iterations) # limit number of simplex iterations

        # define variables ---------------------------------------------------------
        
        # define pulling actions
        to_pull = [model.addVar(vtype=GRB.BINARY, name=f'action_{j}') for j in range(n_arms)]

        # budget constraint
        model.addConstr((gp.quicksum(to_pull) <= budget), 'budget')

        if with_combinatorial:
            unrolled = [[[] for j in range(n_arms)] for k in range(n_arms)]
            for j in range(n_arms):
                for k in range(n_arms):
                    if j==k or (j,k) in edge_list:
                        unrolled[j][k] = [model.addVar(vtype=GRB.BINARY, name=f'unrolled_{j}_{k}_{t}')
                                        for t in range(max_path_length)]
                    else:
                        unrolled[j][k] = [0] * max_path_length

            self.unrolled = unrolled  # keep track of the time-unrolled graph to calclulate path later

            # define constraints -------------------------------------------------------
            
            # begin at source
            model.addConstr((gp.quicksum([unrolled[source][k][0] for k in range(n_arms) 
                                        if self.rmab.has_edge(source, k)]) == 1), 'begin_source')

            # end at source
            model.addConstr((gp.quicksum([unrolled[k][source][-1] for k in range(n_arms) 
                                        if self.rmab.has_edge(k, source)]) == 1), 'end_source')

            # sum of flow is path length
            model.addConstrs(((gp.quicksum([unrolled[j][k][t] for (j,k) in edge_list]) == 1)
                                for t in range(max_path_length)), 'path length')

            # valid path (equal flow in and out)
            model.addConstrs(((gp.quicksum([unrolled[k][j][t] for k in range(n_arms)]) == 
                            gp.quicksum([unrolled[j][k][t+1] for k in range(n_arms)])) 
                            for j in range(n_arms) for t in range(max_path_length-1)), f'equal_flow')
            
            # all pulled arms are visited along path
            model.addConstrs(((to_pull[j] <= gp.quicksum(unrolled[k][j][t] for k in range(n_arms) for t in range(max_path_length)))
                                for j in range(n_arms)), 'pulled_arms_visited')
        
        model.update()
        return model


    def optimize_myopic(self, plot=False):
        if isinstance(self.rmab.rmab, MultiStateRMAB):
            total_time = time.time()
            model = self.get_master_mip(with_combinatorial=True)
            actions = self.get_first_stage_variables(model)

            optimize_multistate_myopic(self.rmab.rmab, model, actions)
    
            try:
                obj_val = model.objVal
                plot = False
                if plot:
                    # calculate & plot actual path
                    path = [self.rmab.source]
                    curr_node = self.rmab.source
                    for t in range(self.rmab.path_length):
                        next_node = np.argmax([self.unrolled[curr_node][k][t].X if self.rmab.has_edge(curr_node, k) else 0 for k in range(self.rmab.n_arms)])
                        path.append(next_node)
                        curr_node = next_node

                    print('   path', path)

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
                'solving_time': model.runtime,
                'model': model
            }
            return results
        
        else:
            return super().optimize()