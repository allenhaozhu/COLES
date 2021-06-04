import numpy as np
import scipy.sparse as sp
import torch
import load_process
import sys
import pickle as pkl
import networkx as nx
import math

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -1.0).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return d_mat_inv_sqrt.dot(adj)

def normalize_adj(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

# def load_adj_neg(num_nodes, sample):
#     '''
#     adj_neg = np.zeros((num_nodes, num_nodes), dtype=float)
#     l = np.random.randint(0, num_nodes, size=num_nodes * (sample + 1))
#     t = 0
#     for i in range(num_nodes):
#         s = 0
#         adj_neg[i, i] = sample
#         while s < sample:
#             if i != l[t]:
#                 adj_neg[i, l[t]] = -1
#                 s += 1
#             t += 1
#     '''
#     col = np.random.randint(0, num_nodes, size=num_nodes * sample)
#     row = np.repeat(range(num_nodes), sample)
#     data = np.ones(num_nodes * sample)
#     adj_neg = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
#     #adj_neg = (sp.eye(adj_neg.shape[0]) * sample - adj_neg).toarray()
#     #adj_neg = (sp.eye(adj_neg.shape[0]) - adj_neg/sample).toarray()
#     #adj_neg = (adj_neg / sample).toarray()
#     adj_neg = normalize_adj(adj_neg).toarray()
#
#     return adj_neg

def load_adj_neg(num_nodes, sample):
    '''
    adj_neg = np.zeros((num_nodes, num_nodes), dtype=float)
    l = np.random.randint(0, num_nodes, size=num_nodes * (sample + 1))
    t = 0
    for i in range(num_nodes):
        s = 0
        adj_neg[i, i] = sample
        while s < sample:
            if i != l[t]:
                adj_neg[i, l[t]] = -1
                s += 1
            t += 1
    '''
    col = np.random.randint(0, num_nodes, size=num_nodes * sample)
    row = np.repeat(range(num_nodes), sample)
    index = np.not_equal(col,row)
    col = col[index]
    row = row[index]
    new_col = np.concatenate((col,row),axis=0)
    new_row = np.concatenate((row,col),axis=0)
    #data = np.ones(num_nodes * sample*2)
    data = np.ones(new_col.shape[0])
    adj_neg = sp.coo_matrix((data, (new_row, new_col)), shape=(num_nodes, num_nodes))
    #adj_neg = (sp.eye(adj_neg.shape[0]) * sample - adj_neg).toarray()
    #adj_neg = (sp.eye(adj_neg.shape[0]) - adj_neg/sample).toarray()
    #adj_neg = (adj_neg / sample).toarray()
    adj_neg = normalize_adj(adj_neg)

    return adj_neg.toarray()

# def load_adj_neg(num_nodes, sample):
#     '''
#     adj_neg = np.zeros((num_nodes, num_nodes), dtype=float)
#     l = np.random.randint(0, num_nodes, size=num_nodes * (sample + 1))
#     t = 0
#     for i in range(num_nodes):
#         s = 0
#         adj_neg[i, i] = sample
#         while s < sample:
#             if i != l[t]:
#                 adj_neg[i, l[t]] = -1
#                 s += 1
#             t += 1
#     '''
#     col = np.random.randint(0, num_nodes, size=num_nodes * sample)
#     row = np.repeat(range(num_nodes), sample)
#     index = np.greater(col,row)
#     col = col[index]
#     row = row[index]
#     new_col = np.concatenate((col,row),axis=0)
#     new_row = np.concatenate((row,col),axis=0)
#     data = np.ones(new_row.shape[0])
#     adj_neg = sp.coo_matrix((data, (new_row, new_col)), shape=(num_nodes, num_nodes))
#     # adj_neg = (sp.eye(adj_neg.shape[0]) * sample - adj_neg)
#     adj_neg = normalize_adj(adj_neg)
#     #adj_neg = (sp.eye(adj_neg.shape[0]) - adj_neg/sample).toarray()
#
#     return adj_neg.toarray()



def load_dataset(dataset_str):
    if dataset_str == 'cora_full':
        data_name = dataset_str + '.npz'
        data_graph = load_process.load_npz_to_sparse_graph("data/{}".format(data_name))
        data_graph.to_undirected()
        data_graph.to_unweighted()
        A = data_graph.adj_matrix
        X = data_graph.attr_matrix
        adj_normalized = torch.from_numpy(normalize_adj(sp.eye(A.shape[0]) + A).toarray()).float()
        X = torch.from_numpy(X.todense()).float()
    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset_str == 'citeseer':
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj_normalized = torch.from_numpy(normalize_adj(sp.eye(adj.shape[0]) + adj).toarray()).float()
        X = torch.from_numpy(features.todense()).float()

    return X, adj_normalized

def load_dataset_adj_lap(dataset_str):
    if dataset_str == 'cora_full':
        data_name = dataset_str + '.npz'
        data_graph = load_process.load_npz_to_sparse_graph("data/{}".format(data_name))
        data_graph.to_undirected()
        data_graph.to_unweighted()
        A = data_graph.adj_matrix
        X = data_graph.attr_matrix
        adj_normalized = torch.from_numpy(normalize_adj(sp.eye(A.shape[0]) + A).toarray()).float()
        X = torch.from_numpy(X.todense()).float()
        Laplacian = torch.from_numpy(normalize_adj(A).toarray()).float()
    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset_str == 'citeseer':
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj_normalized = torch.from_numpy(normalize_adj(sp.eye(adj.shape[0]) + adj).toarray()).float()
        #Laplacian = torch.from_numpy(sp.eye(adj.shape[0]) - normalize_adj(adj).toarray()).float()
        Laplacian = torch.from_numpy(normalize_adj(adj).toarray()).float()
        X = torch.from_numpy(features.todense()).float()

    return X, adj_normalized, Laplacian

def load_reddit_data_lap(data_path="data/", normalization="AugNormAdj", cuda=True):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    train_lap = adj = adj + adj.T
    adj = train_lap + sp.eye(adj.shape[0])
    train_adj = adj[train_index, :][:, train_index]
    train_lap = train_lap[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    adj = aug_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    train_adj = aug_normalized_adjacency(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    train_lap = normalized_adjacency(train_lap)
    train_lap = sparse_mx_to_torch_sparse_tensor(train_lap).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, train_lap, features, labels, train_index, val_index, test_index
