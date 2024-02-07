import torch
import torch.nn as nn
import torch.nn.functional as F

K = 100# find K maximum coefficients




class OursLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(OursLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = 0.2
        self.dropout = dropout

        self.W1 = nn.Parameter(torch.zeros(size=(in_features, out_features)))#(d,d')
        self.W2 = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # (d,d')
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.a3 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a3.data, gain=1.414)
        self.a4 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a4.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.bn3 = nn.BatchNorm1d(out_features)

    def forward(self,Sinput, Rinput, inter_adj,city_adj, province_adj, source_index):

        h1 = torch.mm(Rinput, self.W1)#(M,d')
        h2 = torch.mm(Sinput, self.W2)#(N,d')
        N,M = inter_adj.size()
        #M = h1.size(0)
        B = len(source_index)
        h2_ = h2[source_index]#(B,d')

        inter_input = torch.cat([h1.unsqueeze(0).expand(N, -1, -1), h2.unsqueeze(1).expand(-1, M, -1)], dim=2).view(N, -1, 2 * self.out_features)  # (N,M,2d')
        e12 = F.leaky_relu(torch.matmul(inter_input, self.a).squeeze(2), negative_slope=0.2)  # N*M
        zero_vec_inter = -9e15 * torch.ones_like(e12)
        attention_inter = torch.where(inter_adj > 0, e12, zero_vec_inter)  # (N,M)
        attention_inter = F.softmax(attention_inter, dim=1)
        attention_inter = F.dropout(attention_inter, self.dropout, training=self.training) #(N,M)

        repeat_h3 = h2_.unsqueeze(1).expand(-1, N, -1)
        repeat_h4 = h2_.unsqueeze(1).expand(-1, N, -1)

        city_input = torch.cat([repeat_h3, repeat_h3], dim=2)
        e3 = F.leaky_relu(torch.matmul(city_input, self.a3).squeeze(2), negative_slope=0.2) #(B,N)

        prov_input = torch.cat([repeat_h4, repeat_h4], dim=2)
        e4 = F.leaky_relu(torch.matmul(prov_input, self.a4).squeeze(2), negative_slope=0.2) #(B,N)

        zero_vec34 = -9e15 * torch.ones_like(e3)
        attention3 = torch.where(city_adj[source_index] > 0, e3, zero_vec34)  # (B,N)
        attention4 = torch.where(province_adj[source_index] > 0, e4, zero_vec34)  # (B,N)

        SUM_county = torch.sum(torch.exp(attention3), dim=1, keepdim=True) \
                     + torch.sum(torch.exp(attention4), dim=1, keepdim=True) \
                     + torch.sum(torch.exp(attention_inter[source_index]), dim=1, keepdim=True)
        attention3 = torch.exp(attention3) / SUM_county
        attention3 = F.dropout(attention3, self.dropout, training=self.training)  # (B,N)
        attention4 = torch.exp(attention4) / SUM_county
        attention4 = F.dropout(attention4, self.dropout, training=self.training)  # (B,N)

        InterRC = attention_inter @ h1#(N,M) @ (M,d') = (N,d')
        IntraNC = attention3.t() @ h2_ + attention4.t() @ h2_#(N,B) @ (B,d') = (N,d')
        v_output = self.leakyrelu(self.bn1(attention_inter.t() @ h2))#(M,N) @ (N,d')=(M,d')
        u_output = self.leakyrelu(self.bn2(InterRC + IntraNC))#(N,d')

        #InterRC = (B,M) @ (M,d') = (B,d')
        #IntraNC = (B,N) @ (N,d') = (B,d')
        #v_output = (M,B) @ (B,d') = (M,d')
        #u_output = (B,d')+(B,d¡¯) = (B,d')
        #h = (B,d') @ (d',M) = (B,M)
        h_prime = u_output @ v_output.t()
        return F.elu(h_prime)#(N,M)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.alpha = alpha
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # (D,M)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)  # (N,M)
        N, M = h.size()

        repeat_h = h.unsqueeze(1).expand(-1, M, -1)
        a_input = torch.cat([repeat_h, repeat_h], dim=2)  # (N,M,2d)
        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2), negative_slope=0.2)  # (N,M)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.mul(attention, h)
        return F.elu(h_prime)  # (N,M)


class ablation1(nn.Module):
    def __init__(self, in_features, out_features, n_classes, n_heads, dropout, gdp, Scount, Rcount):
        super(ablation1, self).__init__()
        gdp_values = torch.tensor(list(gdp.values())).view(-1, 1)
        self.Sfeatures = nn.Parameter(torch.cat((torch.rand([Scount, in_features])[:, :-1], gdp_values), dim=1))#(N,d)
        self.Rfeatures = nn.Parameter(torch.rand([Rcount, in_features]))#(M,d)
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.dropout = dropout

        self.attention = OursLayer(in_features, out_features, dropout=dropout)

    def forward(self, inter_adj,city_adj, province_adj, source_index):
        s_input = F.dropout(self.Sfeatures, self.dropout, training=self.training)
        r_input = F.dropout(self.Rfeatures, self.dropout, training=self.training)
        x = self.attention(s_input, r_input, inter_adj, city_adj, province_adj,source_index)#(N,M)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(x)
        return F.log_softmax(x, dim=1)#(N,M)



class OursLayer2(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(OursLayer2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = 0.2
        self.dropout = dropout

        self.W1 = nn.Parameter(torch.zeros(size=(in_features, out_features)))#(d,d')
        self.W2 = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # (d,d')
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.a3 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a3.data, gain=1.414)
        self.a4 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a4.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.bn3 = nn.BatchNorm1d(out_features)

    def forward(self,Sinput, Rinput, inter_adj,city_adj, province_adj, source_index):

        h1 = torch.mm(Rinput, self.W1)#(M,d')
        h2 = torch.mm(Sinput, self.W2)#(N,d')
        N,M = inter_adj.size()
        #M = h1.size(0)
        B = len(source_index)
        h2_ = h2[source_index]#(B,d')

        inter_input = torch.cat([h1.unsqueeze(0).expand(N, -1, -1), h2.unsqueeze(1).expand(-1, M, -1)], dim=2).view(N, -1, 2 * self.out_features)  # (N,M,2d')
        e12 = F.leaky_relu(torch.matmul(inter_input, self.a).squeeze(2), negative_slope=0.2)  # N*M
        zero_vec_inter = -9e15 * torch.ones_like(e12)
        attention_inter = torch.where(inter_adj > 0, e12, zero_vec_inter)  # (N,M)
        attention_inter = F.softmax(attention_inter, dim=1)
        attention_inter = F.dropout(attention_inter, self.dropout, training=self.training) #(N,M)

        repeat_h3 = h2_.unsqueeze(1).expand(-1, N, -1)
        repeat_h4 = h2_.unsqueeze(1).expand(-1, N, -1)

        city_input = torch.cat([repeat_h3, repeat_h3], dim=2)
        e3 = F.leaky_relu(torch.matmul(city_input, self.a3).squeeze(2), negative_slope=0.2) #(B,N)

        prov_input = torch.cat([repeat_h4, repeat_h4], dim=2)
        e4 = F.leaky_relu(torch.matmul(prov_input, self.a4).squeeze(2), negative_slope=0.2) #(B,N)

        zero_vec34 = -9e15 * torch.ones_like(e3)
        attention3 = torch.where(city_adj[source_index] > 0, e3, zero_vec34)  # (B,N)
        attention4 = torch.where(province_adj[source_index] > 0, e4, zero_vec34)  # (B,N)

        attention3 = F.softmax(attention3, dim=1)
        attention3 = F.dropout(attention3, self.dropout, training=self.training)  # (B,N)
        attention4 = F.softmax(attention4, dim=1)
        attention4 = F.dropout(attention4, self.dropout, training=self.training)  # (B,N)

        InterRC = attention_inter @ h1#(N,M) @ (M,d') = (N,d')
        IntraNC = attention3.t() @ h2_ + attention4.t() @ h2_#(N,B) @ (B,d') = (N,d')
        v_output = self.leakyrelu(self.bn1(attention_inter.t() @ h2))#(M,N) @ (N,d')=(M,d')
        u_output = self.leakyrelu(self.bn2(InterRC + IntraNC))#(N,d')

        h_prime = u_output @ v_output.t()
        return F.elu(h_prime)#(N,M)


class ablation2(nn.Module):
    def __init__(self, in_features, out_features, n_classes, n_heads, dropout, gdp, Scount, Rcount):
        super(ablation2, self).__init__()
        gdp_values = torch.tensor(list(gdp.values())).view(-1, 1)
        self.Sfeatures = nn.Parameter(torch.cat((torch.rand([Scount, in_features])[:, :-1], gdp_values), dim=1))#(N,d)
        self.Rfeatures = nn.Parameter(torch.rand([Rcount, in_features]))#(M,d)
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.dropout = dropout

        self.attentions = [OursLayer2(in_features, out_features, dropout=dropout) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(n_classes * n_heads, n_classes, dropout=dropout)

    def forward(self, inter_adj,city_adj, province_adj, source_index, record=False, Coeff12=None, Coeff3=None, Coeff4=None):
        s_input = F.dropout(self.Sfeatures, self.dropout, training=self.training)
        r_input = F.dropout(self.Rfeatures, self.dropout, training=self.training)
        #def forward(self,Sinput, Rinput, inter_adj,city_adj, province_adj, source_index):
        x = torch.cat([att(s_input, r_input, inter_adj, city_adj, province_adj,source_index) for att in self.attentions], dim=1)#(N,M*heads)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, inter_adj))
        return F.log_softmax(x, dim=1)#(N,M)

###################################################

class OursLayer3(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(OursLayer3, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = 0.2
        self.dropout = dropout

        self.W1 = nn.Parameter(torch.zeros(size=(in_features, out_features)))#(d,d')
        self.W2 = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # (d,d')
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.a3 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a3.data, gain=1.414)
        self.a4 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a4.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.bn3 = nn.BatchNorm1d(out_features)

    def forward(self,Sinput, Rinput, inter_adj,city_adj, province_adj, source_index):

        h1 = torch.mm(Rinput, self.W1)#(M,d')
        h2 = torch.mm(Sinput, self.W2)#(N,d')
        N,M = inter_adj.size()

        inter_input = torch.cat([h1.unsqueeze(0).expand(N, -1, -1), h2.unsqueeze(1).expand(-1, M, -1)], dim=2).view(N, -1, 2 * self.out_features)  # (N,M,2d')
        e12 = F.leaky_relu(torch.matmul(inter_input, self.a).squeeze(2), negative_slope=0.2)  # N*M
        zero_vec_inter = -9e15 * torch.ones_like(e12)
        attention_inter = torch.where(inter_adj > 0, e12, zero_vec_inter)  # (N,M)
        attention_inter = F.softmax(attention_inter, dim=1)
        attention_inter = F.dropout(attention_inter, self.dropout, training=self.training) #(N,M)

        v_output = self.leakyrelu(self.bn1(attention_inter.t() @ h2))#(M,N) @ (N,d')=(M,d')
        u_output = self.leakyrelu(self.bn2(attention_inter @ h1))#(N,d')

        h_prime = u_output @ v_output.t()
        return F.elu(h_prime)#(N,M)

class ablation3(nn.Module):
    def __init__(self, in_features, out_features, n_classes, n_heads, dropout, gdp, Scount, Rcount):
        super(ablation3, self).__init__()
        gdp_values = torch.tensor(list(gdp.values())).view(-1, 1)
        self.Sfeatures = nn.Parameter(torch.cat((torch.rand([Scount, in_features])[:, :-1], gdp_values), dim=1))#(N,d)
        self.Rfeatures = nn.Parameter(torch.rand([Rcount, in_features]))#(M,d)
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.dropout = dropout

        self.attentions = [OursLayer3(in_features, out_features, dropout=dropout) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(n_classes * n_heads, n_classes, dropout=dropout)

    def forward(self, inter_adj,city_adj, province_adj, source_index):
        s_input = F.dropout(self.Sfeatures, self.dropout, training=self.training)
        r_input = F.dropout(self.Rfeatures, self.dropout, training=self.training)
        x = torch.cat([att(s_input, r_input, inter_adj, city_adj, province_adj,source_index) for att in self.attentions], dim=1)#(N,M*heads)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, inter_adj))
        return F.log_softmax(x, dim=1)#(N,M)