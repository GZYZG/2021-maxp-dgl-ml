#-*- coding:utf-8 -*-

# Author:james Zhang
"""
    utilities file for Pytorch models
"""

import torch as th
from functools import wraps
import traceback
from _thread import start_new_thread
import torch.multiprocessing as mp

from models import GraphSageModel, GraphConvModel, GraphAttnModel


class early_stopper(object):

    def __init__(self, patience=10, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.best_value = None
        self.is_earlystop = False
        self.count = 0
        self.val_preds = []
        self.val_logits = []

    def earlystop(self, loss, preds, logits):

        value = -loss

        if self.best_value is None:
            self.best_value = value
            self.val_preds = preds
            self.val_logits = logits
        elif value < self.best_value + self.delta:
            self.count += 1
            if self.verbose:
                print('EarlyStoper count: {:02d}'.format(self.count))
            if self.count >= self.patience:
                self.is_earlystop = True
        else:
            self.best_value = value
            self.val_preds = preds
            self.val_logits = logits
            self.count = 0


# According to https://github.com/pytorch/pytorch/issues/17199, this decorator
# is necessary to make fork() and openmp work together.
def thread_wrapped_func(func):
    """
    用于Pytorch的OpenMP的包装方法。Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function


def l1_regularization(model):
    """参数的L1 正则化"""
    def prod(x):
        if len(x) == 0:
            return None
        elif len(x) == 1:
            return x[0]
        else:
            return x[0] * prod(x[1:])
    regularization_loss = 0
    n = 0
    for param in model.parameters():
        regularization_loss += th.sum(abs(param))
        n += prod(param.shape)
    return regularization_loss / n


def create_model(gnn_model, in_feat, hidden_dim, n_layers, n_classes, n_heads=3, dropout=0, feat_drop=0, attn_drop=0):
    """模型工厂"""
    if gnn_model == 'graphsage':
        model = GraphSageModel(in_feat, hidden_dim, n_layers, n_classes, dropout=dropout)
    elif gnn_model == 'graphconv':
        model = GraphConvModel(in_feat, hidden_dim, n_layers, n_classes,
                               norm='both', activation=F.relu, dropout=dropout)
    elif gnn_model == 'graphattn':
        if isinstance(n_heads, int):
            heads = [n_heads] * n_layers
        elif len(n_heads) != n_layers:
            raise ValueError(f"Length of heads{len(heads)} shoud equal n_layers({n_layers})")
                  
        model = GraphAttnModel(in_feat, hidden_dim, n_layers, n_classes,
                               heads=heads, activation=F.relu, feat_drop=feat_drop, attn_drop=feat_drop)
    else:
        raise NotImplementedError('So far, only support three algorithms: GraphSage, GraphConv, and GraphAttn')
        
    return model