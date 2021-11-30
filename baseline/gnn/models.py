#-*- coding:utf-8 -*-

# Author:james Zhang

"""
    Three common GNN models.
"""
import torch as th
import torch.nn as thnn
import torch.nn.functional as F
import dgl.nn as dglnn
import gc


class GraphSageModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 activation=F.relu,
                 dropout=0):
        super(GraphSageModel, self).__init__()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.activation = activation
        self.dropout = thnn.Dropout(dropout)

        self.layers = thnn.ModuleList()

        hidden_dim.insert(0, in_feats)
        hidden_dim.append(self.n_classes)
        # build multiple layers
        for l in range(len(hidden_dim)-1):
            self.layers.append(dglnn.SAGEConv(in_feats=hidden_dim[l], 
                                               out_feats=hidden_dim[l+1],
                                               aggregator_type='mean'))
#         self.layers.append(dglnn.SAGEConv(in_feats=self.in_feats,
#                                           out_feats=self.hidden_dim,
#                                           aggregator_type='mean'))
#                                           # aggregator_type = 'pool'))
#         for l in range(1, (self.n_layers - 1)):
#             self.layers.append(dglnn.SAGEConv(in_feats=self.hidden_dim,
#                                               out_feats=self.hidden_dim,
#                                               aggregator_type='mean'))
#                                               # aggregator_type='pool'))
#         self.layers.append(dglnn.SAGEConv(in_feats=self.hidden_dim,
#                                           out_feats=self.n_classes,
#                                           aggregator_type='mean'))
#                                           # aggregator_type = 'pool'))

    def forward(self, blocks, features):
        h = features

        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l == 0:
                del features
            del block
            gc.collect()
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)

        return h


class GraphConvModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 norm,
                 activation,
                 dropout):
        super(GraphConvModel, self).__init__()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.norm = norm
        self.activation = activation
        self.dropout = thnn.Dropout(dropout)

        self.layers = thnn.ModuleList()
    
        hidden_dim.insert(0, in_feats)
        hidden_dim.append(self.n_classes)
        # build multiple layers
        for l in range(len(hidden_dim)-1):
            self.layers.append(dglnn.GraphConv(in_feats=hidden_dim[l], 
                                               out_feats=hidden_dim[l+1],
                                               norm=self.norm,
                                               activation=self.activation))
            
#         self.layers.append(dglnn.GraphConv(in_feats=self.in_feats,
#                                            out_feats=self.hidden_dim,
#                                            norm=self.norm,
#                                            activation=self.activation,))
#         for l in range(1, (self.n_layers - 1)):
#             self.layers.append(dglnn.GraphConv(in_feats=self.hidden_dim,
#                                                out_feats=self.hidden_dim,
#                                                norm=self.norm,
#                                                activation=self.activation))
#         self.layers.append(dglnn.GraphConv(in_feats=self.hidden_dim,
#                                            out_feats=self.n_classes,
#                                            norm=self.norm,
#                                            activation=self.activation))

    def forward(self, blocks, features):
        h = features

        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l == 0:
                del features
            del block
            gc.collect()
            if l != len(self.layers) - 1:
                h = self.dropout(h)

        return h

    
class Bias(thnn.Module):
    def __init__(self, size):
        super().__init__()

        self.bias = thnn.Parameter(th.Tensor(size))
        self.reset_parameters()

    def reset_parameters(self):
        thnn.init.zeros_(self.bias)

    def forward(self, x):
        return x + self.bias
    

class GraphAttnModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dims,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop
                 ):
        super(GraphAttnModel, self).__init__()
        self.in_feats = in_feats
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.feat_dropout = feat_drop
        self.attn_dropout = attn_drop
        self.activation = activation

        self.convs = thnn.ModuleList()
        self.linears = thnn.ModuleList()
        self.bns = thnn.ModuleList()
        

        # build multiple layers        
        for l in range(self.n_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            in_hidden = heads[l-1] * hidden_dims[l-1] if l >  0 else self.in_feats
            out_hidden = heads[l] * hidden_dims[l] if l < self.n_layers - 1 else heads[l] * n_classes
            
            self.convs.append(dglnn.GATConv(in_feats=in_hidden,
                                             out_feats=hidden_dims[l] if l < self.n_layers - 1 else n_classes,
                                             num_heads=heads[l],
                                             feat_drop=self.feat_dropout,
                                             attn_drop=self.attn_dropout))
            self.linears.append(thnn.Linear(in_hidden, out_hidden, bias=True))
            if l < self.n_layers - 1:
                self.bns.append(thnn.BatchNorm1d(out_hidden))

        self.dropout0 = thnn.Dropout(feat_drop)
        self.dropout = thnn.Dropout(attn_drop)
        self.bias_last = Bias(n_classes)

    def forward(self, blocks, features):
        h = features
        h = self.dropout0(h)

        for i in range(self.n_layers):
#             print(f"{i}: {blocks[0]}\th: {h.shape}")
            conv = self.convs[i](blocks[0], h)
# #             print(f"conv-{i}: {conv.shape}")
            linear = self.linears[i](h[:blocks[0].number_of_dst_nodes()])
#             print(f"linear-{i}: {linear.shape}")
            linear = linear.view(conv.shape)

            h = conv + linear
#             h = conv

            if i < self.n_layers - 1:
                h = h.flatten(1)
                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)
            if i == 0:
                del features
            del blocks[0]
            gc.collect()

        h = h.mean(1)
#         print(f"h: {h.shape}")
        h = self.bias_last(h)

        return h

