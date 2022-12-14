# PPI
import json
from itertools import product
import os.path as osp
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.utils import remove_self_loops
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

import warnings
warnings.filterwarnings("ignore")

# tsv to csv
import pandas as pd
tsv_files = ["biochemists.tsv", "biochemists-nb-coef.tsv",
             "biochemists-nb-predictions.tsv", "biochemists-zinb-coef.tsv",
             "biochemists-zinb-predictions.tsv"]

if osp.exists("data\\biochemists.csv") == False:
    for i in tsv_files:
        tsv_file = "data" + "\\" + i
        print(tsv_file)
        csv_table = pd.read_table(tsv_file, sep='\t')
        csvfile = "data" + "\\" + i[:len(i) - 4] + "." + "csv"
        print(csvfile)
        csv_table.to_csv(csvfile, index=False)

# csv to pt
if osp.exists("data\\biochemists.pt") == False:
    csv_file = "data" + "\\" + "biochemists.csv"
    print(csv_file)
    train = pd.read_csv(csv_file)
    train_tensor = torch.tensor(train.values)

class PPI(InMemoryDataset):

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):

        assert split in ['train', 'val', 'test']

        super().__init__(root, transform, pre_transform, pre_filter)

        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        splits = ['train', 'valid', 'test']
        files = ['feats.npy', 'graph_id.npy', 'graph.json', 'labels.npy']
        return [f'{split}_{name}' for split, name in product(splits, files)]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        pass
        '''
        path = download_url(self.url, self.root)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        '''

    def process(self):
        import networkx as nx
        from networkx.readwrite import json_graph

        for s, split in enumerate(['train', 'valid', 'test']):
            path = osp.join(self.raw_dir, f'{split}_graph.json')
            with open(path, 'r') as f:
                G = nx.DiGraph(json_graph.node_link_graph(json.load(f)))

            x = np.load(osp.join(self.raw_dir, f'{split}_feats.npy'))
            x = torch.from_numpy(x).to(torch.float)

            y = np.load(osp.join(self.raw_dir, f'{split}_labels.npy'))
            y = torch.from_numpy(y).to(torch.float)

            data_list = []
            path = osp.join(self.raw_dir, f'{split}_graph_id.npy')
            idx = torch.from_numpy(np.load(path)).to(torch.long)
            idx = idx - idx.min()

            for i in range(idx.max().item() + 1):
                mask = idx == i

                G_s = G.subgraph(
                    mask.nonzero(as_tuple=False).view(-1).tolist())
                edge_index = torch.tensor(list(G_s.edges)).t().contiguous()
                edge_index = edge_index - edge_index.min()
                edge_index, _ = remove_self_loops(edge_index)

                data = Data(edge_index=edge_index, x=x[mask], y=y[mask])

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
            torch.save(self.collate(data_list), self.processed_paths[s])


pp1enable = 1

if pp1enable == 1:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
    #the required data is present in ppi folder
else:
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')

train_dataset = PPI(path, split='train')
print(train_dataset.data)
print("number of features:",train_dataset.num_features)
print("train_dataset type:", type(train_dataset))

val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
print("train_loader", type(train_loader))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(train_dataset.num_features, 256, heads=4)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(4 * 256, train_dataset.num_classes, heads=6,
                             concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x


import numpy as np

class ZINB(torch.nn.Module):
    def __init__(self):
        super(ZINB, self).__init__()

    def forward(self, x, mean, disp=30,pi=0.5,scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        mean = mean * scale_factor

        t1 = torch.lgamma(torch.tensor(disp + eps)) + torch.lgamma((x + 1.0)) - torch.lgamma((x + disp + eps))
        t2 = (disp + x) * torch.log(torch.tensor(1.0 + (mean / (disp + eps)))) + (x * (torch.log(torch.tensor(disp + eps)) - torch.log(torch.tensor(mean + eps))))
        nb_final = t1 + t2
        nb_case = nb_final - torch.log(abs(torch.tensor(1.0 - pi + eps)))
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge
        result = torch.mean(result)

        return torch.nn.BCEWithLogitsLoss()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
zinb = ZINB()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train():
    global zinb
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss_op=zinb.forward(model(data.x, data.edge_index), data.y)
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    #model.eval()

    ys, preds = [], []
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        ys.append(data.y)
        print(model(data.x, data.edge_index))


        out = model.forward(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


for epoch in range(1, 5):
    loss = train()
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_f1:.4f}, '
          f'Test: {test_f1:.4f}')
