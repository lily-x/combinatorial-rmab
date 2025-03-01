import time

import numpy as np
import torch

import gurobipy as gp
from gurobipy import GRB

from .approximator import Approximator

class RmabApproximator(Approximator):
    def __init__(self, rmab, model_type='NN-E'):
        """ 
        rmab: MultiActionRMAB """
        self.rmab = rmab
        self.model = None
        self.model_type = model_type


    def get_master_mip(self):
        """
        initialize MIP model with first-stage variables and constraints """
        raise NotImplementedError
    

    def get_scenario_embedding(self, n_scenarios, test_set='0'):
        """ Gets the set of scenarios.  """
        if test_set == '0':
            scenario_embedding = self.rmab.observation().reshape(1, -1)
        else:
            raise ValueError('Test set not defined for INVP')

        return scenario_embedding


    def approximate(self,
                    network,
                    mipper,
                    n_scenarios=4,
                    gap=0.02,
                    time_limit=600,
                    threads=1,
                    log_dir=None,
                    test_set="0",
                    scenario_embedding=None,
                    scenario_probs=None,
                    with_combinatorial=False):
        def callback(model, where):
            """ Callback function to log time, bounds, and first stage sol. """
            if where == gp.GRB.Callback.MIPSOL:
                solving_results['time'].append(model.cbGet(gp.GRB.Callback.RUNTIME))
                solving_results['primal'].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST))
                solving_results['dual'].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND))
                solving_results['incumbent'].append(model.cbGetSolution(first_stage_vars))

        total_time = time.time()

        if with_combinatorial: # special option for the RoutingRMAB
            master_mip         = self.get_master_mip(with_combinatorial=True)
        else:
            master_mip         = self.get_master_mip()
        first_stage_vars   = self.get_first_stage_variables(master_mip)

        # add additional axis
        if torch.is_tensor(scenario_embedding):
            scenario_embedding = scenario_embedding.unsqueeze(0)
        else:
            scenario_embedding = np.expand_dims(scenario_embedding, axis=0)
        
        scenario_probs = [1.]
        
        mipper_inst = mipper(master_mip,
                                       first_stage_vars,
                                       network,
                                       scenario_embedding,
                                       scenario_probs=scenario_probs)
        approximator_mip = mipper_inst.get_mip()
        
        solving_results = {'time': [], 'primal': [], 'dual': [], 'incumbent': []}

        approximator_mip.setParam('LogToConsole', 0) # turn off console output
        if log_dir is not None:
            approximator_mip.setParam('LogFile', log_dir)
            # approximator_mip.setParam('OutputFlag', 1) # verbose output
        approximator_mip.setParam('TimeLimit', time_limit)
        approximator_mip.setParam('MipGap', gap)
        approximator_mip.setParam('Threads', threads)
        approximator_mip.optimize(callback)
        total_time = time.time() - total_time

        solving_results['incumbent'] = [dict(x) for x in solving_results['incumbent']]

        
        # calculate results
        try:
            first_stage_sol = self.get_first_stage_solution(approximator_mip)
            obj_val = approximator_mip.objVal

        except:
            print(f'model status {approximator_mip.Status}')
            if approximator_mip.Status == GRB.OPTIMAL:
                print('model optimal')
            elif approximator_mip.Status == GRB.INFEASIBLE:
                print('infeasible')
            elif approximator_mip.Status == GRB.INF_OR_UNBD:
                print('infeasible or unbounded')
            elif approximator_mip.Status == GRB.UNBOUNDED:
                print('unbounded')
            elif approximator_mip.Status == GRB.CUTOFF:
                print('past cutoff')
            elif approximator_mip.Status in [GRB.ITERATION_LIMIT, GRB.NODE_LIMIT, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]:
                print('past iteration/node/time/solution limit')
            
            print(f'  get_first_stage_solution() with state {scenario_embedding} is unsuccessful...')
            
            # pick random action within budget
            first_stage_sol = self.rmab.get_random_action()

            obj_val = -1
            
        results = {
            'time': total_time,
            'predicted_obj': obj_val,
            'sol': first_stage_sol,
            'solving_results': solving_results,
            'solving_time': approximator_mip.runtime
        }

        return results
    

    def get_first_stage_solution(self, model):
        x_sol = np.zeros(self.rmab.action_dim)
        for var in model.getVars():
            if 'action_' in var.varName:
                idx = int(var.varName.split('_')[-1])
                x_sol[idx] = np.abs(var.x)

        return x_sol
    

    def get_first_stage_variables(self, mip):
        vars = {}
        for i in range(self.rmab.action_dim):
            vars[i] = mip.getVarByName(f'action_{i}')
        return vars


