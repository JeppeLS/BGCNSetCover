import numpy as np
import os


def greedy_solve(adj, cost, grasp=False):
    u, s = np.shape(adj)
    solution = []
    c = 0
    elements_covered = set()
    while len(elements_covered) != u:
        scores = cost / np.sum(adj, axis=0)
        if grasp:
            best_s = np.random.choice(range(s), p=1 - (scores / np.sum(scores)))
        else:
            best_s = np.argmin(scores)
        for row in range(u):
            if adj[row, best_s] == 1:
                elements_covered.add(row)
                adj[row, :] = 0
        solution.append(best_s)
        c += cost[best_s]
    return solution, c

def greedy_eval(data_folder):
    results = {}
    for filename in os.listdir(data_folder):
        inst, type = filename.split('_')
        if type == 'matrix.npy':
            print(inst)
            adj = np.load(data_folder + filename)
            cost = np.ones(adj.shape[1])
            sol, cost = greedy_solve(adj, cost)
            results[inst] = cost
    return results
