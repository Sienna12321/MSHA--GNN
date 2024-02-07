import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.alpha = alpha
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))#(D,M)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)#(N,M)
        N,M = h.size()

        repeat_h = h.unsqueeze(1).expand(-1, M, -1)
        a_input = torch.cat([repeat_h, repeat_h], dim=2)#(N,M,2d)
        #a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2), negative_slope=0.2)#(N,M)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.mul(attention, h)
        return F.elu(h_prime)#(N,M)


class GAT(nn.Module):
    def __init__(self, n_features, n_classes, n_heads, dropout, gdp, N):
        super(GAT, self).__init__()
        gdp_values = torch.tensor(list(gdp.values())).view(-1, 1)
        self.features = nn.Parameter(torch.cat((torch.rand([N, n_features])[:, :-1], gdp_values), dim=1))
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(n_features, n_classes, dropout=dropout) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(n_features * n_heads, n_classes, dropout=dropout)

    def forward(self, adj):
        x = F.dropout(self.features, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)#(N,M*heads)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
