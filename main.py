import os
import random
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from matplotlib import rc
from scipy import sparse

from loss.loss import HindsightLoss
from model.model import MultiRankingNetwork
from utils import A_to_adj, reduce_instance, reduce_instance_multi


class SubsetEvaluation:
    def __init__(self, number_of_networks, bn_params, optim_hyperparams, device):
        self.loss_fn = HindsightLoss()
        self.device = device
        self.loss_values = []
        self.val_losses = []
        self.bn_params = bn_params
        self.model = MultiRankingNetwork(number_of_networks=number_of_networks, bn_params=bn_params, device=device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), **optim_hyperparams)

    def train(self, epochs, data_folder, validation=False, val_folder='validation_instances'):
        for epoch in range(epochs):
            print('Starting epoch: ', epoch + 1, ' of ', epochs)
            running_loss = []
            for filename in os.listdir(data_folder):
                inst, type = filename.split('_')
                if type == 'sparsematrix.npz':
                    adj = scipy.sparse.load_npz(data_folder + filename)
                    solution = np.genfromtxt(data_folder + inst + '_solution.csv', delimiter=',', skip_header=1,
                                             usecols=[2], dtype=np.uint8)
                    solution_torch = torch.from_numpy(solution).float().to(self.device)
                    self.optimizer.zero_grad()
                    subset_scores = self.get_scores(adj)
                    loss = self.loss_fn(subset_scores, solution_torch.view_as(subset_scores[:, 0]))
                    loss.backward()
                    self.optimizer.step()
                    running_loss.append(loss.item())
                    del loss, subset_scores
            self.loss_values.append(np.mean(running_loss))

            if validation:
                running_val = []
                for filename in os.listdir(val_folder):
                    inst, type = filename.split('_')
                    if type == 'sparsematrix.npz':
                        with torch.no_grad():
                            adj = scipy.sparse.load_npz(data_folder + filename)
                            solution = np.genfromtxt(data_folder + inst + '_solution.csv', delimiter=',', skip_header=1,
                                                     usecols=[2], dtype=np.uint8)

                            solution_torch = torch.from_numpy(solution).float().to(self.device)
                            subset_scores = self.get_scores(adj)
                            loss = self.loss_fn(subset_scores, solution_torch.view_as(subset_scores[:, 0]))
                            running_val.append(loss.item())
                self.val_losses.append(np.mean(running_val))

    def plot_loss(self, graph_path, show=False):
        font = {'family': 'serif',
                'serif': ['computer modern roman'],
                'size': 16}
        rc('text', usetex=True)
        rc('font', **font)
        plt.plot(list(range(1, len(np.array(self.loss_values)) + 1)), self.loss_values, 'black', linewidth=2,
                 label='Training Loss', linestyle='solid')
        plt.plot(list(range(2, len(self.val_losses) + 1)), self.val_losses[:-1], 'black', linewidth=2,
                 label='Validation Loss', linestyle='dashed')
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        if self.bn_params['network_type'] == 'gat':
            plt.title('Loss for ' + self.bn_params['network_type'].upper() + '-based model with' + '\n' +  '$l=' + str(
                self.bn_params['num_layers']) + "$, $f=" + str(
                self.bn_params['in_channels']) + '$, and $h=' + str(
                self.bn_params['heads']) + '$', fontsize=20)
        else:
            plt.title('Loss for ' + self.bn_params['network_type'].upper() + '-based model with ' + '\n' + '$l=' + str(self.bn_params['num_layers']) + '$ and $f=' + str(
                self.bn_params['in_channels']) + '$', fontsize=20)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fname=graph_path)
        if show:
            plt.show()
        plt.close()

    def eval(self, data_folder, eval_strat, max_files=1000, local_search=False, timeit=False, print_result=True):
        count = 0
        results = {}
        times = {}
        folder = sorted(os.listdir(data_folder))
        for filename in folder:
            inst, type = filename.split('_')
            if type == 'sparsematrix.npz':
                if print_result:
                    print(inst)
                count += 1
                adj = scipy.sparse.load_npz(data_folder + filename)
                cost = np.ones(adj.shape[1])
                time_start = time.time()
                pred_solution = self.eval_instance(adj, eval_strat)
                time_end = time.time() - time_start
                if local_search:
                    sol = [i for i in range(adj.shape[1]) if pred_solution[i] == 1]
                    sol = self.local_search(adj, sol)
                    pred_solution = np.zeros_like(pred_solution)
                    pred_solution[sol] = 1
                pred_cost = pred_solution @ cost
                if print_result:
                    print(pred_cost)
                results[inst] = pred_cost
                times[inst] = time_end
                if count == max_files: break
        if timeit:
            return results, times
        else:
            return results

    def eval_instance(self, adj, eval_strat):
        with torch.no_grad():
            if eval_strat == 'greedy':
                pred_solution = self.eval_instance_greedy(adj)
            elif eval_strat == 'grasp':
                pred_solution = self.eval_instance_grasp(adj)
            elif eval_strat == 'gts':
                pred_solution = self.eval_instance_gts(adj)
            elif eval_strat == 'early_split':
                pred_solution = self.eval_instance_early_split(adj)
            elif eval_strat == 'late_split':
                pred_solution = self.eval_instance_late_split(adj)
            elif eval_strat == 'variable_split':
                pred_solution = self.eval_instance_variable_split(adj)
            elif eval_strat == 'fixed_split':
                pred_solution = self.eval_instance_fixed_split(adj)
            else:
                raise ValueError('The eval strategy ', eval_strat, ' does not exist')
            return pred_solution

    def eval_instance_greedy(self, adj):
        solution_found = False
        pred_solution = np.zeros(adj.shape[1])
        while not solution_found:
            subset_scores = self.get_scores(adj)
            i = 0
            while True:
                idx = torch.argsort(subset_scores, descending=True, dim=0)[i].item()
                if pred_solution[idx] == 1:
                    i += 1
                else:
                    pred_solution[idx] = 1
                    break
            adj = reduce_instance(adj, idx)
            solution_found = adj.size == 0
        return pred_solution

    def eval_instance_grasp(self, adj, percentage=0.01, runs=40):
        best_sol = None
        best_cost = np.infty
        cost = np.ones(adj.shape[1])
        for i in range(runs):
            pred_solution = self.grasp(adj, percentage)
            pred_cost = pred_solution @ cost
            if pred_cost < best_cost:
                best_sol = pred_solution
                best_cost = pred_cost
        return best_sol

    def grasp(self, adj, percentage):
        solution_found = False
        number_of_subsets = adj.shape[1]
        pred_solution = np.zeros(number_of_subsets)
        while not solution_found:
            subset_scores = self.get_scores(adj)
            idxs = torch.argsort(subset_scores, descending=True, dim=0)[0:int(number_of_subsets * percentage),
                   0].tolist()
            subset_scores = subset_scores[idxs].cpu().numpy().flatten()
            idx = np.random.choice(idxs, p=subset_scores / np.sum(subset_scores))
            pred_solution[idx] = 1
            adj = reduce_instance(adj, idx)
            solution_found = adj.size == 0
        return pred_solution

    def eval_instance_fixed_split(self, adj, amount_of_solutions=10):
        solution = None
        partial_solutions = set()
        subset_scores = self.get_scores(adj)
        idxs = torch.argsort(subset_scores, descending=True, dim=0)[0:amount_of_solutions, 0].tolist()
        for i in range(amount_of_solutions):
            partial_solutions.add((idxs[i],))
        while solution is None:
            partial_solutions_ = sorted(list(partial_solutions.copy()))
            for partial_solution in partial_solutions_:
                partial_solutions.remove(partial_solution)
                red_adj = reduce_instance_multi(adj, list(partial_solution))
                if red_adj.size is 0:
                    solution = partial_solution
                    break
                else:
                    subset_scores = self.get_scores(red_adj)
                    idxs = torch.argsort(subset_scores, descending=True, dim=0)[0:amount_of_solutions, 0].tolist()
                    i = 0
                    while len(partial_solutions) < amount_of_solutions:
                        partial_solutions.add(tuple(sorted(partial_solution + (idxs[i],))))
                        i += 1
        sol = np.zeros(adj.shape[1])
        sol[list(solution)] = 1
        return sol

    def eval_instance_early_split(self, adj, amount_of_solutions=10):
        searching = True
        partial_solutions = set()
        best_solution = None
        best_cost = np.infty
        cost = np.ones(adj.shape[1])
        subset_scores = self.get_scores(adj)
        idxs = torch.argsort(subset_scores, descending=True, dim=0)[0:amount_of_solutions, 0].tolist()
        for i in range(amount_of_solutions):
            partial_solutions.add((idxs[i],))
        while searching:
            partial_solution = random.choice(tuple(partial_solutions))
            partial_solutions.remove(partial_solution)
            red_adj = reduce_instance_multi(adj, list(partial_solution))
            cur_sol = self.eval_instance_greedy(red_adj)
            cur_sol[partial_solution] = 1
            cur_cost = cur_sol @ cost
            if cur_cost < best_cost:
                best_solution = cur_sol
                best_cost = cur_cost
            if len(partial_solutions) == 0:
                searching = False
        return best_solution

    def eval_instance_variable_split(self, adj, split_size=2, step_per_split=None):
        if step_per_split is None:
            sol = self.eval_instance_greedy(adj)
            step_per_split = int(np.sum(sol)/4)
        partial_solutions = set()
        best_solution = None

        # Intial split and expand
        subset_scores = self.get_scores(adj)
        idxs = torch.argsort(subset_scores, descending=True, dim=0)[0:split_size, 0].tolist()
        for i in range(split_size):
            partial_solution, is_sol, _ = self.expand((idxs[i],), adj, step_per_split)
            partial_solutions.add(partial_solution)
            if is_sol and (best_solution is None or len(partial_solution) < len(best_solution)):
                best_solution = partial_solution

        # Split and expand until solution found
        while best_solution is None:
            partial_solutions_ = partial_solutions.copy()
            for partial_solution in partial_solutions_:
                partial_solutions.remove(partial_solution)
                red_adj = reduce_instance_multi(adj, partial_solution)
                if red_adj.size == 0:
                    best_solution = partial_solution
                    break
                subset_scores = self.get_scores(red_adj)
                idxs = torch.argsort(subset_scores, descending=True, dim=0)[0:split_size, 0].tolist()
                for i in range(split_size):
                    partial_solution_, is_sol, step_used = self.expand(partial_solution + (idxs[i],),
                                                                      red_adj, step_per_split)
                    if is_sol and (best_solution is None or len(best_solution) > len(partial_solution_)):
                        best_solution = partial_solution_
                        step_per_split = step_used
                    elif best_solution is None:
                        partial_solutions.add(partial_solution_)
        sol = np.zeros(adj.shape[1])
        sol[list(best_solution)] = 1
        return sol

    def expand(self, partial_solution, adj, step_per_split):
        step_used = 1
        adj = reduce_instance_multi(adj, partial_solution)
        if adj.size == 0:
            is_sol = True
        else:
            is_sol = False
        while step_used < step_per_split:
            adj = reduce_instance_multi(adj, partial_solution)
            if adj.size == 0:
                is_sol = True
                break
            else:
                subset_scores = self.get_scores(adj)
                idx = torch.argsort(subset_scores, descending=True, dim=0)[0].item()
                partial_solution = partial_solution + (idx,)
                step_used += 1
        return tuple(sorted(partial_solution)), is_sol, step_used

    def eval_instance_late_split(self, adj, split_size=2, steps_back=8):
        searching = True
        subset_scores = self.get_scores(adj)
        idx = torch.argsort(subset_scores, descending=True, dim=0)[0].item()
        partial_solution = tuple([idx])
        while searching:
            red_adj = reduce_instance_multi(adj, partial_solution)
            if red_adj.size == 0:
                solution = partial_solution
                searching = False
            else:
                subset_scores = self.get_scores(red_adj)
                idx = torch.argsort(subset_scores, descending=True, dim=0)[0].item()
                partial_solution = partial_solution + (idx,)
        solution = list(solution)
        steps_back = min(steps_back, len(solution))
        deconstructed_solution = solution[:-steps_back]
        red_adj = reduce_instance_multi(adj, deconstructed_solution)
        sol = self.eval_instance_variable_split(red_adj, split_size, 1)
        sol[deconstructed_solution] = 1
        return sol

    def eval_instance_gts(self, adj, split_size=2):
        split_prob = 20 / adj.shape[1]
        searching = True
        partial_solutions = set()
        solution = None
        cur_best = np.infty
        subset_scores = self.get_scores(adj)
        idxs = torch.argsort(subset_scores, descending=True, dim=0)[0:split_size, 0].tolist()
        if random.random() < split_prob:
            print('Splitting')
            for i in range(split_size):
                partial_solutions.add((idxs[i],))
        else:
            partial_solutions.add((idxs[0],))
        while searching:
            partial_solution = random.choice(tuple(partial_solutions))
            red_adj = reduce_instance_multi(adj, list(partial_solution))
            if red_adj.size == 0 and cur_best > len(partial_solution):
                solution = partial_solution
                print('New best solution found')
                cur_best = len(solution)
            elif cur_best > len(partial_solution) + 1:
                subset_scores = self.get_scores(red_adj)
                idxs = torch.argsort(subset_scores, descending=True, dim=0)[0:3, 0].tolist()
                if random.random() < split_prob:
                    for i in range(split_size):
                        partial_solutions.add(tuple(sorted(partial_solution + (idxs[i],))))
                else:
                    partial_solutions.add(tuple(sorted(partial_solution + (idxs[0],))))
            partial_solutions.remove(partial_solution)
            if len(partial_solutions) is 0:
                searching = False
        sol = np.zeros(adj.shape[1])
        sol[list(solution)] = 1
        return sol

    def local_search(self, adj, sol):
        solution_found = False
        tabu_list = []
        while not solution_found and len(sol) != len(tabu_list):
            subset_scores = self.get_scores(adj)
            selection = [s for s in sol if s not in tabu_list]
            sorted_scores_idx = torch.argsort(subset_scores[selection], descending=False, dim=0)
            candidate = sorted_scores_idx[0].item()
            for i in range(1, len(sorted_scores_idx)):
                flip = [sol[candidate], sol[sorted_scores_idx[i]]]
                partial_solution = [s for s in sol if s not in flip]
                reduced_graph = reduce_instance_multi(adj, partial_solution)
                any_sol = reduced_graph.sum(axis=0)
                solution_found = int(any_sol.max()) == int(reduced_graph.shape[0])
                if solution_found:
                    idx = np.argmax(any_sol)
                    sol = partial_solution + [idx]
                    print(len(sol))
                    sol = self.local_search(adj, sol)
                    break
                # print('nope')
            tabu_list.append(candidate)

        return sol

    def get_scores(self, adj):
        num_elements, num_subsets = adj.shape
        features = torch.ones((num_elements + num_subsets, self.bn_params['in_channels'])).float().to(self.device)
        sub_to_uni, uni_to_sub = A_to_adj(adj, self.device)
        subset_scores = self.model(features, uni_to_sub.indices(), sub_to_uni.indices(), num_elements,
                                   num_subsets)
        return subset_scores

    def save_model(self, path):
        state_dicts = {}
        state_dicts['model_state_dict'] = self.model.state_dict()
        state_dicts['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dicts['loss_values'] = self.loss_values
        state_dicts['val_losses'] = self.val_losses
        torch.save(state_dicts, path)
        print('Model saved succesfully')

    def load_model(self, path):
        if os.path.isfile(path):
            print('Loading model')
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss_values = checkpoint['loss_values']
            self.val_losses = checkpoint['val_losses']


def get_model_path(num_layers, in_channels, epoch, network_type, heads=1):
    path = os.path.dirname(os.path.realpath(__file__)) + '/checkpoints/' + network_type + '/model_' + str(
        num_layers) + 'layers_' + str(in_channels) + 'features_' + str(epoch) + 'epochs.pt'
    if network_type == 'gat':
        path = os.path.dirname(os.path.realpath(__file__)) + '/checkpoints/' + network_type + '/model_' + str(
            num_layers) + 'layers_' + str(in_channels) + 'features_' + str(epoch) + 'epochs_' + str(heads) + 'heads.pt'
    return path


def get_graph_path(num_layers, in_channels, epoch, network_type, heads=1):
    if network_type == 'gat':
        path = os.path.dirname(os.path.realpath(__file__)) + '/graphs/loss_' + network_type + '_' + str(
        num_layers) + 'layers_' + str(in_channels) + 'features_' + str(epoch) + 'epochs_'+ str(heads) + 'heads.pdf'
    else:
        path = os.path.dirname(os.path.realpath(__file__)) + '/graphs/loss_' + network_type + '_' + str(
        num_layers) + 'layers_' + str(in_channels) + 'features_' + str(epoch) + 'epochs.pdf'
    return path


def read_model(num_layers, in_channels, network_type, heads=1, use_max_epochs=True, use_epochs=None):
    files = []
    folder = os.getcwd() + '/checkpoints/' + network_type + '/'

    for filename in os.scandir(folder):
        split = filename.name.split('_')
        if network_type == 'gat' and str(num_layers) + 'layers' in split and str(in_channels) + 'features' in split and str(heads) + 'heads.pt' in split:
            files.append(filename.name)
        elif network_type != 'gat' and str(num_layers) + 'layers' in split and str(in_channels) + 'features' in split:
            files.append(filename.name)
    files.sort()
    if len(files) == 0:
        warnings.warn('No model found')
        return get_model_path(num_layers, in_channels, 0, network_type)
    if use_max_epochs:
        epochs = []
        for file in files:
            vals = re.findall('\d+', file)
            vals = [int(i) for i in vals]
            epochs.append(vals[2])
        idx = np.argmax(epochs)
    elif use_epochs != None:
        i = 0
        for file in files:
            vals = re.findall('\d+', file)
            vals = [int(i) for i in vals]
            if vals[2] == use_epochs:
                idx = i
            i += 1
    else:
        print('Files found:')
        for i in range(len(files)):
            print(str(i) + ': ', files[i])
        idx = int(input("Choose model to load: "))
    file = folder + files[idx]
    print(file)
    return file
