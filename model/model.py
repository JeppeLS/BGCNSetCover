import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module, Parameter
from torch_geometric.nn import GINConv, GCNConv, GATConv
from torch_geometric.nn.inits import glorot

class MultiRankingNetwork(Module):
    def __init__(self, number_of_networks, bn_params, device):
        super(MultiRankingNetwork, self).__init__()
        self.number_of_networks = number_of_networks
        self.device = device
        for i in range(self.number_of_networks):
            self.add_module(str(i), BipartiteNetwork(**bn_params, device=device))

    def forward(self, features, uni_to_sub, sub_to_uni, num_elements, num_subsets):
        subset_scores = torch.empty((num_subsets, self.number_of_networks)).to(self.device)
        i = 0
        for module in self.children():
            subset_scores[:, i] = module(features, uni_to_sub, sub_to_uni, num_elements, num_subsets).view_as(subset_scores[:, i])
            i += 1
        return subset_scores

class BipartiteNetwork(Module):
    def __init__(self, num_layers, in_channels, network_type, device, heads = 1):
        super(BipartiteNetwork, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.subset_layers = []
        self.universe_layers = []

        for i in range(self.num_layers):
            if i == self.num_layers-1:
                out_channels = 1
            else:
                out_channels = in_channels
            if network_type == 'gin':
                self.subset_layers.append(self._GIN_layer(in_channels,out_channels, hidden_layers=10, hidden_channels=16))
                self.universe_layers.append(self._GIN_layer(in_channels,out_channels, hidden_layers=10, hidden_channels=16))
            elif network_type == 'gcn':
                self.subset_layers.append(GCNConv(in_channels,out_channels, normalize=True).to(device))
                self.universe_layers.append(GCNConv(in_channels,out_channels, normalize=True).to(device))
            elif network_type == 'gat':
                self.subset_layers.append(GATConv(in_channels,out_channels, heads=heads, concat=False).to(device))
                self.universe_layers.append(GATConv(in_channels, out_channels, heads=heads, concat=False).to(device))
            self.add_module('subset_module' + str(i), self.subset_layers[i])
            self.add_module('universe_module' + str(i), self.universe_layers[i])


    def forward(self, features, uni_to_sub, sub_to_uni, num_elements, num_subsets):
        """

        :param features: input features in shape [num_universe + num_subsets, num_features]
        :param uni_to_sub:  directed adjancency with edges from subsets to universe in COO format
        :param sub_to_uni:  directed adjancency with edges from universe to subsets in COO format
        :return:
        """
        for i in range(self.num_layers):
            subset_feats = self.subset_layers[i](x = features, edge_index = uni_to_sub)[num_elements:,:] #adj with edges uni -> sub
            uni_feats = self.universe_layers[i](x = features, edge_index = sub_to_uni)[:num_elements,:]
            features = torch.cat((uni_feats, subset_feats))
            if i < self.num_layers-1:
                features = features.sigmoid()
            else:
                features = features.sigmoid()
        return features[num_elements:,:]

    def _GIN_layer(self, in_channels, out_channels, hidden_layers ,hidden_channels, dropout_rate = 0.3):
        layers = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(nn.Sigmoid())
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_channels,hidden_channels))
            layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_channels, out_channels))
        network = nn.Sequential(*layers).to(self.device)
        return GINConv(nn = network).to(self.device)


