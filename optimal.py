import os
import numpy as np

def optimal_sols(data_folder):
    res = {}
    for filename in os.listdir(data_folder):
        inst, type = filename.split('_')
        if type == 'solution.csv':
            solution = np.genfromtxt(data_folder + filename, delimiter=',', skip_header=1,
                                         usecols=[2], dtype=np.uint8)
            cost = np.ones_like(solution)
            res[inst] = (np.dot(solution, cost))
    return res