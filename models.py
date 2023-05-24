import torch.nn as nn
import math
import torch
import torch.nn.functional as F


class Linear(nn.Module): 
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MLP_encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(MLP_encoder, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        return x

class MLP_classifier(nn.Module):#
    def __init__(self, nfeat, nclass, dropout):
        super(MLP_classifier, self).__init__()
        self.Linear1 = Linear(nfeat, nclass, dropout, bias=True)

    def forward(self, x):
        out = self.Linear1(x)
        return torch.log_softmax(out, dim=1), out

class DDPT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, use_bn = False):
        super(DDPT, self).__init__()

        self.encoder = MLP_encoder(nfeat=nfeat,
                                 nhid=nhid,
                                 dropout=dropout)

        self.classifier = MLP_classifier(nfeat=nhid,
                                         nclass=nclass,
                                         dropout=dropout)

        self.proj_head1 = Linear(nhid, nhid, dropout, bias=True)

        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(nfeat)
            self.bn2 = nn.BatchNorm1d(nhid)

    def forward(self, features, eval = False):
        if self.use_bn:
            features = self.bn1(features)
        query_features = self.encoder(features)
        if self.use_bn:
            query_features = self.bn2(query_features)

        output, emb = self.classifier(query_features)
        if not eval:
            emb = self.proj_head1(query_features)
        return emb, output