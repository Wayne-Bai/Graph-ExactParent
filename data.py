import torch
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from random import shuffle

import networkx as nx
import pickle as pkl
import scipy.sparse as sp
import logging

import random
import shutil
import os
import time
from model import *
from utils import *


def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                #### a wrong example, should not permute here!
                # shuffle(neighbor)
                next = next + neighbor
        output = output + next
        start = next
    return output

def dfs_seq(G, start_id):
    DFS = nx.dfs_tree(G, source=start_id)
    output = list(DFS)
    return output

def generate_Graph(matrix, G:nx.Graph,args):
    # node
    N, _ = matrix.shape
    node_idx, f_dict = [],[]
    for node in range(N):
        indicator = matrix[node, args.max_num_node:]
        if indicator.any():
            node_idx.append(node)
            f_dict.append({f'f{feature_idx}': matrix[node, args.max_num_node+feature_idx] for feature_idx in range(args.max_node_feature_num)})
    node_list = list(zip(node_idx,f_dict))
    G.add_nodes_from(node_list)
    #edge
    for node in range(N):
        indicator = matrix[node, :node]
        if indicator.any():
            for idx in range(node+1):
                if matrix[node, idx] == 1:
                    G.add_edge(node,idx)

    return G

def find_last_element(list):
    ls = [i for i, j in enumerate(list) if j != 0]
    return ls[-1]

########## use pytorch dataloader
class Graph_sequence_sampler_pytorch(torch.utils.data.Dataset):
    def __init__(self, G_list, args, max_num_node=None, max_prev_node=None, iteration=20000):
        self.adj_all, self.node_num_all, self.raw_node_f_all, self.len_all = \
            [], [], [], []
        self.DFS_first_node = []
        self.args = args
        for i, G in enumerate(G_list):
            # add node_type_feature_matrix and edge_type_feature_matrix
            for node in G.nodes():
                if G.nodes[node]['f1'] == 1:
                    first_n = list(G.nodes).index(node)
                    self.DFS_first_node.append(first_n)

            self.adj_all.append(np.array(nx.to_numpy_matrix(G)))

            node_idx_global = np.asarray(list(G.nodes))
            self.node_num_all.append(node_idx_global)

            self.raw_node_f_all.append(dict(G.nodes._nodes))

            self.len_all.append(G.number_of_nodes())

        self.n = max(self.len_all)

        # self.max_prev_node = max_prev_node

        # # sort Graph in descending order
        # len_batch_order = np.argsort(np.array(self.len_all))[::-1]
        # self.len_all = [self.len_all[i] for i in len_batch_order]
        # self.adj_all = [self.adj_all[i] for i in len_batch_order]
    def __len__(self):
        return len(self.adj_all)
    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        node_dict = self.raw_node_f_all[idx].copy()
        node_num_list = self.node_num_all[idx].copy()

        node_feature = self.construct_raw_node_f(node_dict,node_num_list)

        len_batch = adj_copy.shape[0]

        adj_copy_matrix = np.asmatrix(adj_copy)
        G =nx.from_numpy_matrix((adj_copy_matrix))
        start_idx = self.DFS_first_node[idx]
        x_idx = np.array(dfs_seq(G,start_idx))

        node_feature = node_feature[x_idx,:]
        adj_copy = adj_copy[np.ix_(x_idx,x_idx)]
        adj_low_tri = np.tril(adj_copy,k=-1)

        x_batch = np.zeros((self.n+1,self.n+node_feature.shape[1]))
        x_batch[0,:] = 1
        temp_adj = np.zeros((self.n, self.n))
        temp_feature = np.zeros((self.n,node_feature.shape[1]))

        temp_adj[:adj_low_tri.shape[0],:adj_low_tri.shape[1]] = adj_low_tri
        temp_feature[:node_feature.shape[0],:node_feature.shape[1]] = node_feature
        # x_batch[1:node_feature.shape[0]+1,:node_feature.shape[0]+node_feature.shape[1]] = np.concatenate((adj_low_tri,node_feature), axis=1)
        x_batch[1:self.n + 1, :] = np.concatenate(
            (temp_adj,temp_feature), axis=1)

        # x_batch = np.zeros((self.n, node_feature.shape[1] + self.args.max_child_node))
        # x_batch[:node_feature.shape[0], :node_feature.shape[1]+number_of_children.shape[1]] = np.concatenate((node_feature, number_of_children), axis=1)

        return {'x':x_batch,'len':len_batch}

    def construct_raw_node_f(self, node_dict, node_num_list):
        node_attr_list = list(next(iter(node_dict.values())).keys())
        N, NF = len(node_dict), len(node_attr_list)
        offset = min(node_num_list)
        raw_node_f = np.zeros(shape=(N, NF))  # pad 0 for small graphs
        # idx_list = list(range(N))
        for node, f_dict in node_dict.items():
            if node in node_num_list:
                raw_node_f[node - offset] = np.asarray(list(f_dict.values()))  # 0-indexed

        raw_node_f = raw_node_f[node_num_list - offset, :]
        # raw_node_f[:,-1] = 1
        return raw_node_f
    def construct_node_child(self, node_child, node_num_list):
        node_child_num_list = list(node_child.values())
        N,CN = len(node_child), max(node_child_num_list)+1
        offset = min(node_num_list)
        number_of_child = np.zeros(shape=(N,CN))
        for node, child in node_child.items():
            number_of_child[node][child] = 1
        number_of_child = number_of_child[node_num_list - offset, :]
        return number_of_child
