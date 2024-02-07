from __future__ import division
from __future__ import print_function

import time
import argparse
from Ablation import *
import torch.optim as optim
import gc

from model import *
#from Ours import *
from HGANE import *
from torch.utils.data import Dataset, DataLoader, random_split
import dataset
from dataset import *
# Training settings

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--epochs', type=int, default=5,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--K', type=int, default=100,
                    help='K maximum coefficients')
args = parser.parse_args()

######################################
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

        if record:
            #indexes = [i for i in range(N)]
            Coeff12[source_index] = attention_inter[source_index] # whole attention coeff
            Coeff3[source_index] = attention3#find_top_indexes(attention3, indexes, K).to(torch.float) #(B,K) max indexes
            Coeff4[source_index] = attention4#find_top_indexes(attention4, indexes, K).to(torch.float)  # (B,K) max indexes

        InterRC = attention_inter @ h1#(N,M) @ (M,d') = (N,d')
        IntraNC = attention3.t() @ h2_ + attention4.t() @ h2_#(N,B) @ (B,d') = (N,d')
        v_output = self.leakyrelu(self.bn1(attention_inter.t() @ h2))#(M,N) @ (N,d')=(M,d')
        u_output = self.leakyrelu(self.bn2(InterRC + IntraNC))#(N,d')

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


class Ours(nn.Module):
    def __init__(self, in_features, out_features, n_classes, n_heads, dropout, gdp, Scount, Rcount):
        super(Ours, self).__init__()
        gdp_values = torch.tensor(list(gdp.values())).view(-1, 1)
        self.Sfeatures = nn.Parameter(torch.cat((torch.rand([Scount, in_features])[:, :-1], gdp_values), dim=1))#(N,d)
        self.Rfeatures = nn.Parameter(torch.rand([Rcount, in_features]))#(M,d)
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.dropout = dropout

        self.attentions = [OursLayer(in_features, out_features, dropout=dropout) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(n_classes * n_heads, n_classes, dropout=dropout)

    def forward(self, inter_adj,city_adj, province_adj, source_index):
        s_input = F.dropout(self.Sfeatures, self.dropout, training=self.training)
        r_input = F.dropout(self.Rfeatures, self.dropout, training=self.training)
        #def forward(self,Sinput, Rinput, inter_adj,city_adj, province_adj, source_index):
        x = torch.cat([att(s_input, r_input, inter_adj, city_adj, province_adj,source_index) for att in self.attentions], dim=1)#(N,M*heads)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, inter_adj))
        return F.log_softmax(x, dim=1)#(N,M)




Dataset = dataset.HigherDataset()
num_samples = len(Dataset)
train_size = int(0.9 * len(Dataset))
test_size = len(Dataset) - train_size

time1 = time.time()
train_dataset,test_dataset = random_split(Dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
Scount, Rcount = Dataset.get_count()
inter_adj,city_adj, province_adj = Dataset.get_adjacent()#(N,M)
inter_adj = normalize_adjacency_matrix(inter_adj)
city_adj = normalize_adjacency_matrix(city_adj)
province_adj = normalize_adjacency_matrix(province_adj)
GDP = Dataset.get_gdp()
time2 = time.time()
print("load data: {}".format(time2-time1))

# Model and optimizer
#model = GCN(nfeat=64,nhid=args.hidden,nclass=Rcount,dropout=args.dropout, gdp=GDP,N = Scount)
#model = GAT(n_features=32, n_classes=Rcount, n_heads=2, dropout=args.dropout,gdp=GDP, N=Scount)
#model = GraphAttentionLayer(in_features=64, out_features=64, Scount=Scount, Rcount=Rcount, gdp=GDP,dropout=0.5)
t3 = time.time()

#originally 128, 128 in and out features
model = ablation3(in_features=128, out_features=64, n_classes=Rcount, n_heads=2, dropout=0.5, gdp=GDP, Scount=Scount, Rcount=Rcount)
optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=3e-4)
t4 = time.time()
print('load model: {}'.format(t4-t3))

model = model.to(device)
inter_adj = inter_adj.to(device)
city_adj = city_adj.to(device)
province_adj = province_adj.to(device)

def train(epoch):
    t = time.time()
    model.train()
    Loss_train = 0
    for i, data in enumerate(train_loader, 0):
        source_index, recipient_index = data
        source_index = source_index.to(device)
        recipient_index = recipient_index.to(device)

        optimizer.zero_grad()
        output = model(inter_adj,city_adj, province_adj, source_index)
        #loss_train = F.nll_loss(output, recipient_index)
        loss_train = F.nll_loss(output[source_index], recipient_index)
        Loss_train += loss_train.item()
        loss_train.backward()
        optimizer.step()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(Loss_train/len(train_loader)),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()

    y_pred_test = []
    y_true_test = []
    y_preds = []
    Loss_test = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            source_index, recipient_index = data
            source_index = source_index.to(device)
            recipient_index = recipient_index.to(device)
            output = model(inter_adj,city_adj, province_adj, source_index)
            #loss_test = F.nll_loss(output, recipient_index)
            loss_test = F.nll_loss(output[source_index], recipient_index)
            Loss_test += loss_test.item()

            preds = output[source_index]
            pred_label = preds.max(1)[1].type_as(recipient_index)
            for tag in pred_label:
                y_preds.append(tag.cpu().item())
            for pred in preds:
                y_pred_test.append(pred.detach().cpu().numpy().tolist())
            for t in recipient_index:
                y_true_test.append(t.cpu().item())
        y_preds = np.array(y_preds)
        y_pred_test = np.array(y_pred_test)
        y_true_test = np.array(y_true_test)
        auc_test = calculate_auc(y_pred_test, y_true_test)
        acc = calculate_accuracy(y_preds, y_true_test)
        precision_macro, recall_macro = calculate_precision_recall(y_preds, y_true_test,'macro')
        f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro)
        precision_micro, recall_micro = calculate_precision_recall(y_preds, y_true_test, 'micro')
        f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)
        print("Test set results:",
              "loss= {:.4f}".format(Loss_test/len(test_loader)),
              "auc= {:.4f}".format(auc_test),
              "accuracy= {:.4f}".format(acc),
              "precision_macro= {:.4f}".format(precision_macro),
              "recall_macro= {:.4f}".format(recall_macro),
              "f1_macro= {:.4f}".format(f1_macro),
              "precision_micro= {:.4f}".format(precision_micro),
              "recall_micro= {:.4f}".format(recall_micro),
              "f1_micro= {:.4f}".format(f1_micro),)

def Record():
    model.eval()
    with torch.no_grad():
        int_tensor = torch.arange(Scount)
        chunked_tensors = int_tensor.split(args.batch_size)
        for source_index in chunked_tensors:
            source_index = source_index.to(device)
            model(inter_adj,city_adj, province_adj, source_index)


record = False
# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
    test()
'''
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Recording attention parameters
del Dataset
gc.collect()
torch.cuda.empty_cache()
time.sleep(180)

#global Coeff12
Coeff12 = torch.zeros([Scount,Rcount]).to(device)
Coeff3 = torch.zeros([Scount,Scount]).to(device)
Coeff4 = torch.zeros([Scount,Scount]).to(device)
record = True
Record() # save the attention parameter

Coeff12, Coeff3, Coeff4 = Coeff12.cpu().numpy(), Coeff3.cpu().numpy(), Coeff4.cpu().numpy()
attCoeff = {'Coeff12': Coeff12, 'Coeff3': Coeff3, 'Coeff4': Coeff4}
print(Coeff12)
file_path = '/data/home/mengtong_zhang/impoverished_students/anonymous_data/' + year + 'AttCoeff.npz'
np.savez(file_path, **attCoeff)

'''
