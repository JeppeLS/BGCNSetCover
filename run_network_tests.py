import os
import re

import scipy.sparse
import torch
from greedy import greedy_eval
from main import SubsetEvaluation, get_model_path, read_model, get_graph_path
from optimal import optimal_sols
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_models(layers, features, network_type, epochs, data_folder, val_folder, head_list=None):
    for layer in layers:
        for feature in features:
            if head_list is not None:
                for heads in head_list:
                    train_model(layer, feature, network_type, epochs, data_folder, val_folder, heads=heads)
            else:
                train_model(layer,feature, network_type, epochs, data_folder, val_folder)

def eval_models(layers, features, network_type, test_folder, head_list=None, use_max_epochs=True, epochs=None):
    for layer in layers:
        for feature in features:
            if head_list is not None:
                for heads in head_list:
                    eval_model(layer, feature, network_type, test_folder, heads, use_max_epochs, epochs)
            else:
                eval_model(layer, feature, network_type, test_folder, use_max_epochs=use_max_epochs, epochs=epochs)



def eval_model(layer, feature, network_type, test_folder, heads=1, use_max_epochs=True, epochs=None):
    bn_params = {
        'num_layers': layer,
        'in_channels': feature,
        'network_type': network_type
    }
    if network_type == 'gat':
        bn_params['heads'] = heads
    optim_hyperparams = {
        'lr': 0.001,
        'betas': (0.9, 0.999),
    }
    model = SubsetEvaluation(number_of_networks=1,
                             bn_params=bn_params,
                             optim_hyperparams=optim_hyperparams,
                             device=device)
    model_path = read_model(**bn_params, use_max_epochs=use_max_epochs, use_epochs=epochs)
    model.load_model(model_path)
    print('Evaling model with ' + str(layer) + ' layer and ' + str(feature) + ' features, trained for ' + str(epochs) + ' epochs.')
    print('Model is of type ' + str(network_type) + (' with ' + (str(heads) + ' heads') if network_type == 'gat' else ''))
    results = model.eval(data_folder=test_folder, eval_strat='greedy', print_result=False)
    optimal = optimal_sols(test_folder)
    diff = []
    for k1, v1 in results.items():
        for k2, v2 in optimal.items():
            if k1 == k2:
                diff.append((v1-v2)/v2)
    print('Difference to CPLEX: ' + str(np.mean(diff)))



def train_model(layers, feature, network_type, epochs, data_folder, val_folder, heads=1):
    bn_params = {
        'num_layers': layers,
        'in_channels': feature,
        'network_type': network_type,
        'heads': heads
    }
    optim_hyperparams = {
        'lr': 0.001,
        'betas': (0.9, 0.999),
    }
    model = SubsetEvaluation(number_of_networks=1,
                             bn_params=bn_params,
                             optim_hyperparams=optim_hyperparams,
                             device=device)
    model_path = read_model(**bn_params, use_max_epochs=True)
    model.load_model(model_path)
    model.train(epochs=epochs, data_folder=data_folder, validation=True, val_folder=val_folder)
    model_path = get_model_path(**bn_params, epoch=len(model.loss_values))
    model.save_model(model_path)

def make_plots():
    network_types = ['gcn', 'gin', 'gat']
    for network_type in network_types:
        for file in os.scandir(os.getcwd() + '/checkpoints/' + network_type + '/'):
            name = file.name
            vals = re.findall('\d+', name)
            vals = [int(i) for i in vals]
            if network_type == 'gat':
                bn_params = {'num_layers': vals[0], 'in_channels': vals[1], 'network_type': network_type, 'heads': vals[3]}
            else:
                bn_params = {'num_layers': vals[0], 'in_channels': vals[1], 'network_type': network_type}
            optim_hyperparams = {
                'lr': 0.001,
                'betas': (0.9, 0.999),
            }
            graph_path = get_graph_path(epoch=vals[2], **bn_params)
            if os.path.exists(graph_path) and network_type != 'gat':
                continue
            else:
                print(graph_path)
            model = SubsetEvaluation(number_of_networks=1,
                                     bn_params=bn_params,
                                     optim_hyperparams=optim_hyperparams,
                                     device=device)
            model.load_model(get_model_path(epoch=vals[2], **bn_params))
            model.plot_loss(graph_path)

def beasley_test(num_layers, in_channels, network_type, use_max_epochs=True, use_epochs=None, eval_strat='greedy', heads=1, local_search = False):
    data_folder1 = os.getcwd() + '/scp_non_unicost/'
    data_folder2 = os.getcwd() + '/scp_unicost/'
    bn_params = {'num_layers': num_layers, 'in_channels': in_channels, 'network_type': network_type}
    if network_type == 'gat':
        bn_params['heads'] = heads
    optim_hyperparams = {
        'lr': 0.001,
        'betas': (0.9, 0.999),
    }
    model = SubsetEvaluation(number_of_networks=1,
                             bn_params=bn_params,
                             optim_hyperparams=optim_hyperparams,
                             device=device)
    model.load_model(read_model(**bn_params, use_max_epochs=use_max_epochs, use_epochs=use_epochs))
    results1, time1 = model.eval(data_folder=data_folder1, eval_strat=eval_strat, local_search=local_search, timeit=True)
    results2, time2 = model.eval(data_folder=data_folder2, eval_strat=eval_strat, timeit=True)
    if network_type == 'gat':
        name = eval_strat + '_' + str(num_layers) + 'layers_' + str(in_channels) + 'features_' + str(
            len(model.loss_values)) + 'epochs_' + network_type + '_' + str(heads) + 'heads.txt'
        timepath = 'time_' + name
        path = 'results_' + name
    else:
        name = eval_strat + '_' + str(num_layers) + 'layers_' + str(in_channels) + 'features_' + str(
            len(model.loss_values)) + 'epochs_' + network_type + '.txt'
        timepath = 'time_' + name
        path = 'results_' + name
    if local_search:
        path = 'local_search_' + path
    with open(timepath, 'w') as f:
        print(time1, file=f)
        print(time2, file=f)
    with open(path, 'w') as f:
        print(results1, file=f)
        print(results2, file=f)


def greedy_test():
    data_folder1 = os.getcwd() + '/scp_non_unicost/'
    data_folder2 = os.getcwd() + '/scp_unicost/'

    results1 = greedy_eval(data_folder=data_folder1)
    results2 = greedy_eval(data_folder=data_folder2)
    with open('results_greedy.txt', 'w') as f:
        print(results1, file=f)
        print(results2, file=f)

def toy_example(num_layers, in_channels, network_type, use_max_epochs=True, use_epochs=None, eval_strat='greedy', heads=1, local_search = False):
    a = np.array([[1,1,0],
                  [1,0,1],
                  [1,1,0],
                  [1,0,1],
                  [0,1,0],
                  [0,0,1]])
    a = scipy.sparse.coo_matrix(a)
    bn_params = {'num_layers': num_layers, 'in_channels': in_channels, 'network_type': network_type}
    if network_type == 'gat':
        bn_params['heads'] = heads
    optim_hyperparams = {
        'lr': 0.001,
        'betas': (0.9, 0.999),
    }
    model = SubsetEvaluation(number_of_networks=1,
                             bn_params=bn_params,
                             optim_hyperparams=optim_hyperparams,
                             device=device)
    model.load_model(read_model(**bn_params, use_max_epochs=use_max_epochs, use_epochs=use_epochs))
    print('Predicted solution is: ' + str(model.eval_instance(a, eval_strat)))


if __name__ == '__main__':
    data_folder = os.getcwd() + '/generated_instances/'
    val_folder = os.getcwd() + '/validation_instances/'
    test_folder = os.getcwd() + '/test_instances/'
    eval_strats = ['greedy','early_split', 'variable_split', 'late_split', 'fixed_split']

    #Uncomment here to make loss plots for all models saved in checkpoint folder
    #make_plots()

    #Uncomment to perform number of layers test for GCN
    #layers = [5,10,15,30]
    #features = [16]
    #eval_models(layers,features,'gcn', test_folder,use_max_epochs=False, epochs=90)

    #Uncomment here to perform the feature test for GCN
    #layers = [15]
    #features = [8,16,32]
    #eval_models(layers,features,'gcn', test_folder,use_max_epochs=False, epochs=90)

    #Uncomment here to perform the GAT test:
    #layers = [5,10,15]
    #features = [16]
    #head_list = [1,2,4]
    #eval_models(layers,features,'gat',test_folder,head_list, use_max_epochs=False, epochs=90)

    #Uncomment here to perform the Toy Example given in the thesis
    #toy_example(15,16,'gcn', use_max_epochs=False,use_epochs=90,eval_strat='greedy')

    #Uncomment here to benchmark on beasley instance for GCN based model with 15 layers and 16 features
    for eval_strat in eval_strats:
        beasley_test(15, 16, 'gcn', eval_strat=eval_strat, use_max_epochs=False, use_epochs=90)

    #Uncomment here to benchmark on beasley instance for GAT based model with 5 layers and 16 features and 2 heads
    #for eval_strat in eval_strats:
    #    beasley_test(5, 16, 'gat', heads=2, eval_strat=eval_strat, use_max_epochs=False, use_epochs=90)

    #Uncomment here to benchmark on beasley instance for GAT based model with 10 layers and 16 features and 2 heads
    #for eval_strat in eval_strats:
    #    beasley_test(10, 16, 'gat', heads=2, eval_strat=eval_strat, use_max_epochs=False, use_epochs=90)