import numpy as np
import scipy
import torch
from scipy import sparse
from torch_geometric.nn import GCNConv

def A_to_adj(A, device):
    """
    Creates two directed adjencency matrice.
    Uni_to_sub
      U    S
    U 0    A
    S 0    0

    Sub_to_uni
      U    S
    U 0    0
    S A.t  0
    """
    num_elements, num_subsets = A.shape
    values = A.data
    indices = np.vstack((A.row, A.col))
    indices[1, :] += num_elements

    i = torch.LongTensor(indices)
    v = torch.LongTensor(values)
    uni_to_sub = torch.sparse.LongTensor(i, v, torch.Size([num_elements + num_subsets, num_elements + num_subsets])).to(
        device).coalesce()
    sub_to_uni = uni_to_sub.transpose(0, 1).coalesce()
    return sub_to_uni, uni_to_sub

def check_sol(adj, solution):
    res = (adj @ solution) >= 1
    return np.all(res)

def reduce_instance(A, subset):
    element_to_remove, _ = A.getcol(subset).nonzero()
    idx = [i for i in range(A.shape[0]) if i not in element_to_remove]
    csr = scipy.sparse.csr_matrix(A)
    return sparse.coo_matrix(csr[idx, :])


def reduce_instance_multi(A, subsets):
    csr = scipy.sparse.csr_matrix(A)
    row_sum = csr[:, subsets].sum(axis=1)
    elements_to_keep = [i for i in range(A.shape[0]) if row_sum[i] == 0]
    return sparse.coo_matrix(csr[elements_to_keep, :])
