import torch
import os.path as osp
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
import torch_geometric.transforms as T
from torch_geometric.utils import dropout_adj, index_to_mask
from ogb.nodeproppred import PygNodePropPredDataset

def split(y, num_classes, train_per_class=20, val_per_class=30):

    indices = []

    for i in range(num_classes):
        index = (y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:train_per_class] for i in indices], dim=0)
    val_index = torch.cat([i[train_per_class:train_per_class+val_per_class] for i in indices], dim=0)
    test_index = torch.cat([i[train_per_class+val_per_class:] for i in indices], dim=0)

    return train_index, val_index, test_index

def get_feature_mask(rate, n_nodes, n_features):
    return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes, n_features)).bool()

def load_data(dset, train_per_class=20, val_per_class=30,
              missing_link=0.0, missing_feature=0.0, ogb_train_ratio=0.01,
              normalize_features=True, use_public_split=False):

    path = osp.join('.', 'Data', dset)
    if dset in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, dset)
    if dset in ['photo', 'computers']:
        dataset = Amazon(path, dset)
    elif dset in ['physics', 'cs']:
        dataset = Coauthor(path, dset)
    elif dset in ['arxiv']:
        dataset = PygNodePropPredDataset('ogbn-'+dset, path)
    else:
        assert Exception

    data = dataset[0]

    if normalize_features:
        data.transform = T.NormalizeFeatures()

    if dset in ['cora', 'citeseer', 'pubmed'] and use_public_split:
        print('Using public split of {}! 20 per class/30 per class/1000 for train/val/test.'.format(dset))
    elif dset not in ['arxiv']:
        train_index, val_index, test_index = split(data.y, dataset.num_classes, train_per_class, val_per_class)
        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)
    else:
        ogb_split = dataset.get_idx_split()
        train_index, val_index, test_index = ogb_split['train'], ogb_split['valid'], ogb_split['test']
        train_index = train_index[torch.randperm(train_index.size(0))]
        train_index = train_index[:int(data.num_nodes * ogb_train_ratio)]
        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)
        data.y = data.y.squeeze(1)

    if missing_link > 0.0:
        print(data.num_edges)
        data.edge_index, _ = dropout_adj(data.edge_index, p=missing_link, force_undirected=True, num_nodes=data.num_nodes)
        print(data.num_edges)

    if missing_feature > 0.0:
        print(torch.sum(data.x))
        feature_mask = get_feature_mask(rate=missing_feature, n_nodes=data.num_nodes,
                                        n_features=data.num_features)
        data.x[~feature_mask] = 0.0
        print(torch.sum(data.x))

    meta = {'num_classes': dataset.num_classes}

    return data, meta








