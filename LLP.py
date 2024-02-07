import argparse
from torch.nn import BCELoss
from torch.nn.functional import cosine_similarity
from model import *
from torch.utils.data import Dataset, DataLoader, random_split
import dataset

parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--link_batch_size', type=int, default=64*1024)
parser.add_argument('--node_batch_size', type=int, default=64*1024)
parser.add_argument('--lr', type=float, default=0.005)#maybe 0.001 better
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--eval_steps', type=int, default=5)
parser.add_argument('--predictor', type=str, default='mlp')  ##inner/mlp
parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
parser.add_argument('--metric', type=str, default='Hits@20', choices=['auc', 'hits@20', 'hits@50'], help='main evaluation metric')
parser.add_argument('--use_valedges_as_input', action='store_true')
parser.add_argument('--True_label', default=10, type=float) #true_label loss
parser.add_argument('--KD_f', default=0.1, type=float) #Representation-based matching KD
parser.add_argument('--KD_p', default=100, type=float) #logit-based matching KD
parser.add_argument('--margin', default=0.1, type=float) #margin for rank-based kd
parser.add_argument('--rw_step', type=int, default=3) # nearby nodes sampled times
parser.add_argument('--ns_rate', type=int, default=1) # # randomly sampled rate over # nearby nodes
parser.add_argument('--hops', type=int, default=2) # random_walk step for each sampling time
parser.add_argument('--ps_method', type=str, default='nb') # positive sampling is rw or nb
parser.add_argument('--batch_size', type=int, default=4096,help='batch size')

args = parser.parse_args()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def KD_cosine(s, t):
    return 1-cosine_similarity(s, t.detach(), dim=-1).mean()
class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        norm_type="none",
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, feats):
        h = feats#feats = (N,d)
        for l, layer in enumerate(self.layers):
            h = layer(h)#(N,d)
            if l != self.num_layers - 1:
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h

class LinkPredictor(torch.nn.Module):
    def __init__(self, predictor, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.predictor = predictor
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        if self.predictor == 'mlp':
            for lin in self.lins[:-1]:
                x = lin(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            #x = self.lins[-1](x)
        elif self.predictor == 'inner':
            x = torch.sum(x, dim=-1)

        return torch.sigmoid(x)

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
        #gdp_values = torch.tensor(list(gdp.values())).view(-1, 1)
        #self.features = nn.Parameter(torch.cat((torch.rand([N, n_features])[:, :-1], gdp_values), dim=1))
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(n_features, n_classes, dropout=dropout) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(n_features * n_heads, n_classes, dropout=dropout)

    def forward(self, input, adj):
        x = F.dropout(input, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)#(N,M*heads)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class Teacher_LinkPredictor(torch.nn.Module):
    def __init__(self, predictor, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(Teacher_LinkPredictor, self).__init__()

        self.predictor = predictor
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        if self.predictor == 'mlp':
            for lin in self.lins[:-1]:
                x = lin(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            #x = self.lins[-1](x)
        elif self.predictor == 'inner':
            x = torch.sum(x, dim=-1)
        return torch.sigmoid(x)

Dataset = dataset.HigherDataset()
num_samples = len(Dataset)
train_size = int(0.9 * len(Dataset))
test_size = len(Dataset) - train_size

train_dataset,test_dataset = random_split(Dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
Scount, Rcount = Dataset.get_count()
adj_inter, adj_intra = Dataset.get_adjacent()#(N,M)
adj_inter = normalize_adjacency_matrix(adj_inter)
adj_intra = normalize_adjacency_matrix(adj_intra)
adj_inter = adj_inter.to(device)
adj_intra = adj_intra.to(device)
GDP = Dataset.get_gdp()
in_features = args.hidden_channels

def train(model, predictor, teacher_model, teacher_predictor, optimizer, args):
    model.train()
    predictor.train()
    mse_loss = torch.nn.MSELoss()
    total_loss =  0

    for i, data in enumerate(train_loader, 0):
        source_index, recipient_index = data
        source_index = source_index.to(device)
        recipient_index = recipient_index.to(device)
        gdp_values = torch.tensor(list(GDP.values())).view(-1, 1)
        features = nn.Parameter(torch.cat((torch.rand([Scount, in_features])[:, :-1], gdp_values), dim=1)).to(device)
        optimizer.zero_grad()

        h = model(features)  # input = (B,d),h = (B,d')
        t_h = teacher_model(features, adj_inter)# t_h = (B,d'), d'=M
        output = predictor(h[source_index],h[recipient_index]).squeeze()#(B,1)

        label_loss = F.nll_loss(output, recipient_index)
        t_out = teacher_predictor(t_h[source_index],t_h[recipient_index]).squeeze().detach()
        loss = args.True_label * label_loss + args.KD_f * KD_cosine(h[source_index],
                                                                    t_h[source_index]) + args.KD_p * mse_loss(output, t_out)
        #lable_loss: -0.7123 KD_cosine: 0.0038 mse_loss: 0.0835
        mse = mse_loss(output, t_out).item()
        loss.backward()
        #print('lable_loss: {:.4f}'.format(label_loss),'KD_cosine: {:.4f}'.format(KD_cosine(h[source_index],t_h[source_index])),'mse_loss: {:.4f}'.format(mse))

        optimizer.step()
        total_loss += loss.item()
    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(total_loss / len(train_loader)))
    return features

def test(model, predictor,features):
    model.eval()
    predictor.eval()
    y_pred_test = []
    y_true_test = []
    y_preds = []
    Loss_test = 0
    with torch.no_grad():
        h = model(features)
        for i, data in enumerate(test_loader, 0):
            source_index, recipient_index = data
            source_index = source_index.to(device)
            recipient_index = recipient_index.to(device)
            preds = predictor(h[source_index],h[recipient_index])#(B,d')

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
        precision_macro, recall_macro = calculate_precision_recall(y_preds, y_true_test, 'macro')
        precision_micro, recall_micro = calculate_precision_recall(y_preds, y_true_test, 'micro')
        print("Test set results:",
              "auc= {:.4f}".format(auc_test),
              "accuracy= {:.4f}".format(acc),
              "precision_macro= {:.4f}".format(precision_macro),
              "recall_macro= {:.4f}".format(recall_macro),
              "precision_micro= {:.4f}".format(precision_micro),
              "recall_micro= {:.4f}".format(recall_micro), )



model = MLP(args.num_layers, args.hidden_channels, args.hidden_channels, args.hidden_channels, args.dropout).to(device)
#input = (n_samples, input_dim), output = (n_samples, output_dim); input_size = input_dim
predictor = LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1,args.num_layers, args.dropout).to(device)
teacher_model = GAT(n_features=Rcount, n_classes=Rcount, n_heads=2, dropout=args.dropout,gdp=GDP, N=Scount).to(device)
teacher_predictor = Teacher_LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

model.reset_parameters()
predictor.reset_parameters()
optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)

for epoch in range(1, 1 + args.epochs):
    features = train(model, predictor, teacher_model, teacher_predictor, optimizer, args)
    test(model, predictor, features)

