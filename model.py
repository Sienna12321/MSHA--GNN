import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.rand([in_features, out_features]))#Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.tensor(out_features,dtype=torch.float32))#torch.FloatTensor
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self,input,  adj):
        #input = (N,d)
        support = input @ self.weight#(N,d) @ (d,d') -> (N,d')
        output = adj.transpose(0, 1) @ support#(M,N) @ (B, d') -> (M,d') torch.spmm
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,gdp, N):
        super(GCN, self).__init__()
        gdp_values = torch.tensor(list(gdp.values())).view(-1, 1)
        self.features = nn.Parameter(torch.cat((torch.rand([N, nfeat])[:, :], gdp_values), dim=1))
        self.gc1 = GraphConvolution(nfeat+1, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self,  adj):
        x = F.relu(self.gc1(self.features, adj))#(M,nhid)
        x = F.dropout(x, self.dropout, training=self.training)#(M,nhid)
        x = F.relu(self.gc2(x, adj.t()))
        #x = F.dropout(x, self.dropout, training=self.training)#(N,nhid)
        #x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

def calculate_auc(y_pred, y_true):
    y_true_binary = label_binarize(y_true, classes=np.unique(y_true))

    aucs = []
    for i in range(y_true_binary.shape[1]):

        auc = roc_auc_score(y_true_binary[:, i], y_pred[:, i])
        aucs.append(auc)

    mean_auc = np.mean(aucs)

    return mean_auc



from sklearn.metrics import accuracy_score

def calculate_accuracy(predicted_labels, true_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

from sklearn.metrics import precision_score, recall_score

def calculate_precision_recall(predicted_labels, true_labels, model):
    precision = precision_score(true_labels, predicted_labels, average=model, zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average=model, zero_division=1)
    return precision, recall


def normalize_adjacency_matrix(adjacency_matrix):
    degrees = torch.sum(adjacency_matrix, dim=0)
    d_sqrt_inv = torch.pow(degrees, -0.5)
    d_sqrt_inv = torch.diag(d_sqrt_inv)
    normalized_adjacency_matrix = torch.mm(torch.mm(adjacency_matrix, d_sqrt_inv), d_sqrt_inv)
    return normalized_adjacency_matrix