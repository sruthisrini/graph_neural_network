#PPI
import json
import os
import os.path as osp
from itertools import product
import tensorflow as tf

import numpy as np
import torch

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.utils import remove_self_loops

import os.path as osp
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

#tsv to csv

import pandas as pd

tsv_files=["biochemists.tsv","biochemists-nb-coef.tsv",
           "biochemists-nb-predictions.tsv","biochemists-zinb-coef.tsv",
           "biochemists-zinb-predictions.tsv"]

if osp.exists("data\\biochemists.csv")==False:
    for i in tsv_files:
        tsv_file="data"+"\\"+i
        print(tsv_file)
        csv_table=pd.read_table(tsv_file,sep='\t')
        csvfile="data"+"\\"+i[:len(i)-4]+"."+"csv"
        print(csvfile)
        csv_table.to_csv(csvfile,index=False)

#csv to pt

if osp.exists("data\\biochemists.pt")==False:
        csv_file="data"+"\\"+"biochemists.csv"
        print(csv_file)
        train = pd.read_csv(csv_file)
        train_tensor = torch.tensor(train.values)


# class Dataset
#class InMemoryDataset(Dataset)

value=10

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

                data = Data(s=edge_index, x=x[mask], y=y[mask])

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
            torch.save(self.collate(data_list), self.processed_paths[s])
pp1enable=1

if pp1enable==1:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
else:
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
print(path)
train_dataset = PPI(path, split='train')

print(train_dataset.data)


print("train_dataset",type(train_dataset))

val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
print("train_loader",type(train_loader))

val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        global two_fifty_Six
        super().__init__()
        self.conv1 = GATConv(train_dataset.num_features, value, heads=4)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, 4 * value)
        self.conv2 = GATConv(4 * value, value, heads=4)
        self.lin2 = torch.nn.Linear(4 * value, 4 * value)
        self.conv3 = GATConv(4 * value, train_dataset.num_classes, heads=6,
                             concat=False)
        self.lin3 = torch.nn.Linear(4 * value, train_dataset.num_classes)


    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x


import statsmodels.api as sm
import numpy as np
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_array_equal,
    assert_equal,
)

def _nan2zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

def _nan2inf(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x)+np.inf, x)

def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.math.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)


def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(x), nelem)


def mse_loss(y_true, y_pred):
    ret = tf.square(y_pred - y_true)

    return _reduce_mean(ret)


# In the implementations, I try to keep the function signature
# similar to those of Keras objective functions so that
# later on we can use them in Keras smoothly:
# https://github.com/fchollet/keras/blob/master/keras/objectives.py#L7
def poisson_loss(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    # we can use the Possion PMF from TensorFlow as well
    # dist = tf.contrib.distributions
    # return -tf.reduce_mean(dist.Poisson(y_pred).log_pmf(y_true))

    nelem = _nelem(y_true)
    y_true = _nan2zero(y_true)

    # last term can be avoided since it doesn't depend on y_pred
    # however keeping it gives a nice lower bound to zero
    ret = y_pred - y_true*tf.math.log(y_pred+1e-10) + tf.math.lgamma(y_true+1.0)

    return tf.divide(tf.reduce_sum(ret), nelem)

class NB(object):
    def __init__(self, theta=None, masking=False, scope='nbinom_loss/',
                 scale_factor=1.0, debug=False):

        # for numerical stability
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor

            if self.masking:
                nelem = _nelem(y_true)
                y_true = _nan2zero(y_true)

            # Clip theta
            theta = tf.minimum(self.theta, 1e6)

            t1 = tf.math.lgamma(theta+eps) + tf.math.lgamma(y_true+1.0) - tf.math.lgamma(y_true+theta+eps)
            t2 = (theta+y_true) * tf.math.log(1.0 + (y_pred/(theta+eps))) + (y_true * (tf.math.log(theta+eps) - tf.math.log(y_pred+eps)))

            if self.debug:
                assert_ops = [
                        tf.verify_tensor_all_finite(y_pred, 'y_pred has inf/nans'),
                        tf.verify_tensor_all_finite(t1, 't1 has inf/nans'),
                        tf.verify_tensor_all_finite(t2, 't2 has inf/nans')]

                tf.summary.histogram('t1', t1)
                tf.summary.histogram('t2', t2)

                with tf.control_dependencies(assert_ops):
                    final = t1 + t2

            else:
                final = t1 + t2

            final = _nan2inf(final)

            if mean:
                if self.masking:
                    final = tf.divide(tf.reduce_sum(final), nelem)
                else:
                    final = tf.reduce_mean(final)


        return final
class ZINB(NB):
    def __init__(self, pi=3, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        super().__init__(scope=scope, **kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            # reuse existing NB neg.log.lik.
            # mean is always False here, because everything is calculated
            # element-wise. we take the mean only in the end
            nb_case = super().loss(y_true, y_pred, mean=False) - tf.math.log(1.0-self.pi+eps)

            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor
            theta = tf.minimum(self.theta, 1e6)

            zero_nb = tf.pow(theta/(theta+y_pred+eps), theta)
            zero_case = -tf.math.log(self.pi + ((1.0-self.pi)*zero_nb)+eps)
            result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
            ridge = self.ridge_lambda*tf.square(self.pi)
            result += ridge

            if mean:
                if self.masking:
                    result = _reduce_mean(result)
                else:
                    result = tf.reduce_mean(result)

            result = _nan2inf(result)

            if self.debug:
                tf.summary.histogram('nb_case', nb_case)
                tf.summary.histogram('zero_nb', zero_nb)
                tf.summary.histogram('zero_case', zero_case)
                tf.summary.histogram('ridge', ridge)

        return result

    def loss_calc(self):
        zero_nb = tf.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -tf.math.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)


using_PPI=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
if using_PPI==1:
    loss_op = torch.nn.BCEWithLogitsLoss()
else:
    #pass
    zinb=ZINB()
    train_dataset = torch.as_tensor(train_dataset, device='cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


for epoch in range(1, 101):
    loss = train()
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_f1:.4f}, '
          f'Test: {test_f1:.4f}')