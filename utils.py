import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from collections import Counter
from pathlib import Path

import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import numpy as np
import sys

input_data = ['cora','citeseer','pubmed',]

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data( dataset_str, data_seed ):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    parent_path = Path(__file__).resolve().parents[1]
    for i in range(len(names)):
        with open("//content/drive/My Drive/gcn/gcn/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("/content/drive/My Drive/gcn/gcn/data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    _nxgraph = nx.from_dict_of_lists( graph )
    adj = nx.adjacency_matrix( _nxgraph )

    adj = adj.astype(np.float32)
    features = features.tocsr()
    features = features.astype(np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val   = range(len(y), len(y)+500)

    n=features.shape[0]-1
    idx_val = range(int(0.2*n))
    idx_train = range((int(0.2*n)), (int(0.2*n))*2)
    idx_test = range((int(0.2*n))*2, n)
    

    idx_val = range(542)
    idx_train = range(542, 542+542)
    idx_test = range(542+542, 2708)

    train_mask = sample_mask( idx_train, labels.shape[0] )
    val_mask   = sample_mask( idx_val,   labels.shape[0] )
    test_mask  = sample_mask( idx_test,  labels.shape[0] )

    print( '#nodes', features.todense().shape[0] ,'#features',features.todense().shape[1])

    print( 'train:valid:test={}:{}:{}'.format( train_mask.sum(), val_mask.sum(), test_mask.sum() ) )

    return adj, features, labels, train_mask, val_mask, test_mask

def sparse_to_tuple( sparse_mx ):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def construct_feed_dict( features, support, labels, labels_mask, placeholders,
                          dropout,alfa,beta ):
    """Construct feed dictionary."""

    feed_dict = dict()
    feed_dict.update( {placeholders['features'].indices: features[0]} )
    feed_dict.update( {placeholders['features'].values:  features[1]} )
    feed_dict.update( {placeholders['support'][i]: _sup for i, _sup in enumerate(support) } )
    feed_dict.update( {placeholders['labels']: labels} )
    feed_dict.update( {placeholders['labels_mask']: labels_mask} )
    feed_dict.update( {placeholders['dropout']: dropout } )
    feed_dict.update( {placeholders['alfa']: alfa } )
    feed_dict.update( {placeholders['beta']: beta } )

    return feed_dict

def construct_feed_dict2( features, support, labels, labels_mask, placeholders,
                          dropout ):
    """Construct feed dictionary."""

    feed_dict = dict()
    feed_dict.update( {placeholders['features'].indices: features[0]} )
    feed_dict.update( {placeholders['features'].values:  features[1]} )
    feed_dict.update( {placeholders['support'][i]: _sup for i, _sup in enumerate(support) } )
    feed_dict.update( {placeholders['labels']: labels} )
    feed_dict.update( {placeholders['labels_mask']: labels_mask} )
    feed_dict.update( {placeholders['dropout']: dropout } )


    return feed_dict
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1), dtype=float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv[np.isnan(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    features.eliminate_zeros()

    return sparse_to_tuple( features )

def sym_normalize_adj( adj ):
    """Symmetrically normalize adjacency matrix."""

    adj = sp.coo_matrix( adj )
    degree = np.array( adj.sum(1) ).flatten()
    d_inv_sqrt = np.power( np.maximum( degree, np.finfo(float).eps ), -0.5 )
    d_mat_inv_sqrt = sp.diags( d_inv_sqrt )
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def row_normalize_adj( adj ):
    '''row normalize adjacency matrix'''

    adj = sp.coo_matrix( adj )
    degree = np.array( adj.sum(1) ).flatten()
    d_mat_inv = sp.diags( 1 / np.maximum( degree, np.finfo(float).eps ) )
    return d_mat_inv.dot( adj ).tocoo()

def preprocess_adj( adj, selfloop_weight=1 ):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""

    return sym_normalize_adj( adj + selfloop_weight * sp.eye(adj.shape[0]) )

def preprocess_high_order_adj( adj, order, eps ):
    adj = row_normalize_adj( adj )

    adj_sum = adj
    cur_adj = adj
    for i in range( 1, order ):
        cur_adj = cur_adj.dot( adj )
        adj_sum += cur_adj
    adj_sum /= order

    adj_sum.setdiag( 0 )
    adj_sum.data[adj_sum.data<eps] = 0
    adj_sum.eliminate_zeros()

    adj_sum += sp.eye( adj.shape[0] )
    return sym_normalize_adj( adj_sum + adj_sum.T )
