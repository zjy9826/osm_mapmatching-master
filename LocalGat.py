import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 可以不用动 W相当于对输入的向量进一步，其实这一步在nettraj里面已经做了。  a相当于nettraj里的W
        # self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(in_features+out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h_i, h_j):
        attention = []
        # print(f"attention layers h_j:{h_j.shape}")
        for item in h_j:
            # print(f"attention layers item:{item.shape}")
            # print(f"attention layers h_i:{h_i.shape}")
            # Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
            wh = torch.concat((h_i,item),dim=0)
            # print(f"wh shape:{wh.shape}")
            e = self._prepare_attentional_mechanism_input(wh)
            attention.append(e)
        # zero_vec = -9e15*torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.tensor(attention,device="cuda")
        # print(f"after attention shape:{attention.shape}")
        attention = F.softmax(attention, dim=0)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h_j)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        # Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        # Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # # broadcast add
        # e = Wh1 + Wh2.T
        e = torch.matmul(Wh, self.a)
        # print(e.shape)
        # return self.leakyrelu(e)
        return e

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class LocalGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(LocalGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nclass * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, h_i, h_j):
        # print(f"h_i:{h_i.shape}")
        # print(f"h_j:{h_j.shape}")
        """
        attention layers h_j:torch.Size([3, 256])
        attention layers item:torch.Size([256])
        """
        h_i = F.dropout(h_i, self.dropout, training=self.training)
        h_j = F.dropout(h_j, self.dropout, training=self.training)

        x = torch.cat([att(h_i, h_j) for att in self.attentions], dim=0)

        # print(f"head x shape:{x.shape}")

        x = F.dropout(x, self.dropout, training=self.training)
        
        # x = F.elu(self.out_att(x, h_j))  后面修改的。

        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=0)