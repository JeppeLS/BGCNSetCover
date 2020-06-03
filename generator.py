import numpy as np


def _generate_instance(u, s, p):
    adj = np.random.choice([0, 1], replace=True, size=(u, s), p=[1 - p, p])
    vertix_nb = np.sum(adj, axis=1)
    for i in range(u):
        if vertix_nb[i] == 0:
            adj[i, np.random.randint(0, s)] = 1
    subset_degree = np.sum(adj, axis=0)
    for j in range(s):
        if subset_degree[j]==0:
            adj[np.random.randint(0, u), j] = 1
            subset_degree[j] = 1
    cost = np.ones(s)
    return adj, cost


dir = './test_instances/'
for i in range(200):
    p = np.random.uniform(0.01, 0.2)
    adj, cost = _generate_instance(200, 500, p)
    np.save(dir+'inst'+str(i)+'_matrix', adj)
    np.save(dir+'inst'+str(i)+'_cost', cost)
