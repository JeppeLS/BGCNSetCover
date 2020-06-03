import numpy as np
import docplex.mp.model as cpx
import pandas as pd
import os


class cplexSolver:
    def solve(self, adj, cost=None):
        u, s = np.shape(adj)
        U = range(1, u + 1)
        S = range(1, s + 1)
        opt_model = cpx.Model(name="SCP Model")
        x_vars = {i: opt_model.binary_var(name="x_{0}".format(i)) for i in S}
        constraints = {j: opt_model.add_constraint(ct=opt_model.sum(adj[j - 1, i - 1] * x_vars[i] for i in S) >= 1,
                                                   ctname="constraint_{0}".format(j)) for j in U}
        if cost is not None:
            objective = opt_model.sum(cost[i - 1] * x_vars[i] for i in S)
        else:
            objective = opt_model.sum(x_vars[i] for i in S)
        time_lim = 10
        opt_model.minimize(objective)
        opt_model.parameters.timelimit.set(time_lim)
        opt_model.parameters.mip.strategy.lbheur.set(1)
        opt_model.parameters.mip.cuts.zerohalfcut.set(2)
        opt_model.solve()
        opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns=["variable_object"])
        #opt_df.reset_index(inplace=True)
        opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.solution_value)
        return opt_df


cplex = cplexSolver()
dir = './test_instances/'

with os.scandir(dir) as files:
    i = 0
    for file in files:
        instance_name, type = file.name.split('_', 2)
        if type == 'matrix.npy':
            print('Solving: ', instance_name)
            i += 1
            print('Iteration: ', i)
            if os.path.isfile(dir+instance_name+'_solution.csv'):
                continue
            adj = np.load(dir + instance_name + '_' + type)
            cost = np.load(dir + instance_name + '_cost.npy')
            cplex_sol_df = cplex.solve(adj, cost)
            cplex_sol_df.to_csv(dir+instance_name+'_solution.csv')
