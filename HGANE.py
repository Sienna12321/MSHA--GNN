import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import sys
import random


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, Scount, Rcount, gdp,dropout=0.5):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = 0.2
        gdp_values = torch.tensor(list(gdp.values())).view(-1, 1)
        self.features = nn.Parameter(torch.cat((torch.rand([Scount, self.in_features])[:, :-1], gdp_values), dim=1))
        self.source_embedding = nn.Parameter(torch.rand([Scount, self.in_features]))
        self.recipient_embedding = nn.Parameter(torch.rand([Rcount, self.in_features]))

        self.W1 = nn.Linear(in_features, out_features, bias=False)#d*d'
        self.W2 = nn.Linear(in_features, out_features, bias=False)#d*d'
        self.a12 = nn.Linear(2*out_features, 1, bias=False)
        self.a3 = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        # ³õÊ¼»¯™àÖØ
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.a12.weight)
        nn.init.xavier_uniform_(self.a3.weight)

    def forward(self, adj_inter, adj_intra,source_index):
        adj_intra = adj_intra[source_index[:, None], source_index]
        adj_inter = adj_inter[source_index]
        h1 = self.W1(self.recipient_embedding)#M*d'
        h2 = self.W2(self.source_embedding[source_index])#N*d'

        M = h1.size()[0]
        N = h2.size()[0]

        inter_input = torch.cat([h1.unsqueeze(0).expand(N, -1, -1), h2.unsqueeze(1).expand(-1, M, -1)], dim=2).view(N, -1, 2 * self.out_features)  # (N,M,2d')
        e12 = self.leakyrelu(self.a12(inter_input).squeeze(2))  # N*M

        repeat_h2 = h2.unsqueeze(1).expand(-1, N, -1)

        intra_input = torch.cat([repeat_h2, repeat_h2.transpose(0, 1)], dim=2)
        e3 = self.leakyrelu(self.a3(intra_input).squeeze(2))

        zero_vec_inter = -9e15 * torch.ones_like(e12)
        zero_vec_intra = -9e15 * torch.ones_like(e3)

        attention_inter = torch.where(adj_inter > 0, e12, zero_vec_inter)#(N,M)
        attention_intra = torch.where(adj_intra > 0, e3, zero_vec_intra)#(N,N)

        #softmax
        SUM_county = torch.sum(torch.exp(attention_intra), dim=1, keepdim=True)\
                     +torch.sum(torch.exp(attention_inter), dim=1, keepdim=True)

        attention_intra = torch.exp(attention_intra) / SUM_county
        attention_intra = F.dropout(attention_intra, self.dropout, training=self.training)  # (N,N)

        SUM_school = torch.sum(torch.exp(attention_inter),dim=1,keepdim=True)
        attention_inter = torch.exp(attention_inter) / SUM_school
        attention_inter = F.dropout(attention_inter, self.dropout, training=self.training) #(N,M)

        u_output = self.leakyrelu(self.bn1(
            self.W1(attention_inter @ self.recipient_embedding) + self.W2(attention_intra @ self.source_embedding[source_index])))#(N,d')
        v_output = self.leakyrelu(self.bn2(self.W1(attention_inter.t() @ self.source_embedding[source_index])))#(M,d')

        h_prime = u_output @ v_output.t()
        return F.elu(h_prime)
