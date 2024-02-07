from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import *
from Ours import *
from HGANE import *
from torch.utils.data import Dataset, DataLoader, random_split
import dataset
# Training settings
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch size')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

class GraphSAGE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, gdp):
        super(GraphSAGE, self).__init__()
        gdp_values = torch.tensor(list(gdp.values())).view(-1, 1)
        self.Sfeatures = nn.Parameter(
            torch.cat((torch.rand([Scount, in_features])[:, :-1], gdp_values), dim=1))  # (N,d)
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, source_index, adj):
        #x=(N,d)
        x = self.Sfeatures[source_index]
        x = F.relu(self.linear1(x))#x=(N,dhid)
        x = torch.mul(adj[source_index], x)  # (N,nhid)
        x = F.relu(self.linear2(x))
        return F.log_softmax(x, dim=1)

Dataset = dataset.HigherDataset_()
num_samples = len(Dataset)
print(num_samples)
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

model = GraphSAGE(in_features=32, hidden_features=32, out_features=Rcount,gdp = GDP)
t3 = time.time()
optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
t4 = time.time()
print('load model:'.format(t4-t3))

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
        output = model(source_index,inter_adj)
        loss_train = F.nll_loss(output, recipient_index)
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
            output = model(source_index,inter_adj)
            loss_test = F.nll_loss(output, recipient_index)
            Loss_test += loss_test.item()

            preds = output
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
        precision_micro, recall_micro = calculate_precision_recall(y_preds, y_true_test, 'micro')
        print("Test set results:",
              "loss= {:.4f}".format(Loss_test/len(test_loader)),
              "auc= {:.4f}".format(auc_test),
              "accuracy= {:.4f}".format(acc),
              "precision_macro= {:.4f}".format(precision_macro),
              "recall_macro= {:.4f}".format(recall_macro),
              "precision_micro= {:.4f}".format(precision_micro),
              "recall_micro= {:.4f}".format(recall_micro))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
    test()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

