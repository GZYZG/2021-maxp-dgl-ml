#-*- coding:utf-8 -*-

"""
    Utilities to handel graph data
"""

import os
import dgl
import pickle
import numpy as np
import torch as th


def load_dgl_graph(base_path):
    """
    读取预处理的Graph，Feature和Label文件，并构建相应的数据供训练代码使用。

    :param base_path:
    :return:
    """
    graphs, _ = dgl.load_graphs(os.path.join(base_path, 'graph.bin'))
    graph = graphs[0]
    print('################ Graph info: ###############')
    print(graph)

    with open(os.path.join(base_path, 'labels.pkl'), 'rb') as f:
        label_data = pickle.load(f)

    labels = th.from_numpy(label_data['label'])
    tr_label_idx = label_data['tr_label_idx']
    val_label_idx = label_data['val_label_idx']
    test_label_idx = label_data['test_label_idx']
    print('################ Label info: ################')
    print('Total labels (including not labeled): {}'.format(labels.shape[0]))
    print('               Training label number: {}'.format(tr_label_idx.shape[0]))
    print('             Validation label number: {}'.format(val_label_idx.shape[0]))
    print('                   Test label number: {}'.format(test_label_idx.shape[0]))

    # get node features
    features = np.load(os.path.join(base_path, 'features.npy'))
    node_feat = th.from_numpy(features).float()
    print('################ Feature info: ###############')
    print('Node\'s feature shape:{}'.format(node_feat.shape))

    return graph, labels, tr_label_idx, val_label_idx, test_label_idx, node_feat


def time_diff(t_end, t_start):
    """
    计算时间差。t_end, t_start are datetime format, so use deltatime
    Parameters
    ----------
    t_end
    t_start

    Returns
    -------
    """
    diff_sec = (t_end - t_start).seconds
    diff_min, rest_sec = divmod(diff_sec, 60)
    diff_hrs, rest_min = divmod(diff_min, 60)
    return (diff_hrs, rest_min, rest_sec)


def CAN(y_pred, prior, k=3, tau=0.9, y_true=None):
    """
    Classification with Alternating Normalization
    reference: 
        1. https://arxiv.org/abs/2109.13449
        2. https://mp.weixin.qq.com/s/3mIipreGJgrl2WuTpfTl6A 
    """
    # 预测结果，计算修正前准确率
    if y_true is not None:
        acc_original = np.mean([y_pred.argmax(1) == y_true])
        print('original acc: %s' % acc_original)

    # 评价每个预测结果的不确定性
#     k = 3
    y_pred_topk = np.sort(y_pred, axis=1)[:, -k:]
    y_pred_topk /= y_pred_topk.sum(axis=1, keepdims=True)
    y_pred_uncertainty = -(y_pred_topk * np.log(y_pred_topk)).sum(1) / np.log(k)

    # 选择阈值，划分高、低置信度两部分
    threshold = tau
    y_pred_confident = y_pred[y_pred_uncertainty < threshold]
    y_pred_unconfident = y_pred[y_pred_uncertainty >= threshold]
    y_true_confident = y_true[y_pred_uncertainty < threshold]
    y_true_unconfident = y_true[y_pred_uncertainty >= threshold]

    # 显示两部分各自的准确率
    # 一般而言，高置信度集准确率会远高于低置信度的
    acc_confident = (y_pred_confident.argmax(1) == y_true_confident).mean()
    acc_unconfident = (y_pred_unconfident.argmax(1) == y_true_unconfident).mean()
    print('confident acc: %s' % acc_confident)
    print('unconfident acc: %s' % acc_unconfident)

    # 从训练集统计先验分布
#     prior = np.zeros(num_classes)
#     for d in train_data:
#         prior[d[1]] += 1.

#     prior /= prior.sum()

    # 逐个修改低置信度样本，并重新评价准确率
    right, alpha, iters = 0, 1, 1
    for i, y in enumerate(y_pred_unconfident):
        Y = np.concatenate([y_pred_confident, y[None]], axis=0)
        for j in range(iters):
            Y = Y**alpha
            Y /= Y.sum(axis=0, keepdims=True)
            Y *= prior[None]
            Y /= Y.sum(axis=1, keepdims=True)
        y = Y[-1]
        if y.argmax() == y_true_unconfident[i]:
            right += 1

    # 输出修正后的准确率
    acc_final = (acc_confident * len(y_pred_confident) + right) / len(y_pred)
    print('new unconfident acc: %s' % (right / (i + 1.)))
    print('final acc: %s' % acc_final)