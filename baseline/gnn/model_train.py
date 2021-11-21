#-*- coding:utf-8 -*-

# Author:james Zhang
"""
    Minibatch training with node neighbor sampling in multiple GPUs
"""

import os
import sys
import argparse
import datetime as dt
import numpy as np
import torch as th
import torch.nn as thnn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import dgl
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader

from models import GraphSageModel, GraphConvModel, GraphAttnModel
from utils import load_dgl_graph, time_diff, train_val_split
from model_utils import early_stopper, thread_wrapped_func, l1_regularization

import logging


writer: SummaryWriter = None
exp_dir: str = ""

def set_logging(log_dir, log_file_name=""):
    assert log_file_name
    logging.basicConfig(filename=os.path.join(log_dir, log_file_name), level=logging.INFO, filemode='a',
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    if not any([h.stream is sys.stdout for h in logging.getLogger().handlers]):
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        

def set_summary_writer(log_dir, log_file_name=""):
    assert log_file_name
    global writer
    writer = SummaryWriter(os.path.join(log_dir, log_file_name))
    

def load_subtensor(node_feats, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = node_feats[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def cleanup():
    dist.destroy_process_group()


def cpu_train(graph_data,
              gnn_model,
              hidden_dim,
              n_layers,
              n_classes,
              fanouts,
              batch_size,
              device,
              num_workers,
              epochs,
              out_path):
    """
        运行在CPU设备上的训练代码。
        由于比赛数据量比较大，因此这个部分的代码建议仅用于代码调试。
        有GPU的，请使用下面的GPU设备训练的代码来提高训练速度。
    """
    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data

    sampler = MultiLayerNeighborSampler(fanouts)
    train_dataloader = NodeDataLoader(graph,
                                      train_nid,
                                      sampler,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=num_workers)

    # 2 initialize GNN model
    in_feat = node_feat.shape[1]

    if gnn_model == 'graphsage':
        model = GraphSageModel(in_feat, hidden_dim, n_layers, n_classes)
    elif gnn_model == 'graphconv':
        model = GraphConvModel(in_feat, hidden_dim, n_layers, n_classes,
                               norm='both', activation=F.relu, dropout=0)
    elif gnn_model == 'graphattn':
        model = GraphAttnModel(in_feat, hidden_dim, n_layers, n_classes,
                               heads=([5] * n_layers), activation=F.relu, feat_drop=0, attn_drop=0)
    else:
        raise NotImplementedError('So far, only support three algorithms: GraphSage, GraphConv, and GraphAttn')

    model = model.to(device)

    # 3 define loss function and optimizer
    loss_fn = thnn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # 4 train epoch
    avg = 0
    iter_tput = []
    start_t = dt.datetime.now()

    print('Start training at: {}-{} {}:{}:{}'.format(start_t.month,
                                                     start_t.day,
                                                     start_t.hour,
                                                     start_t.minute,
                                                     start_t.second))

    for epoch in range(epochs):

        for step, (input_nodes, seeds, mfgs) in enumerate(train_dataloader):

            start_t = dt.datetime.now()

            batch_inputs, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device)
            mfgs = [mfg.to(device) for mfg in mfgs]

            batch_logit = model(mfgs, batch_inputs)
            loss = loss_fn(batch_logit, batch_labels)
            pred = th.sum(th.argmax(batch_logit, dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            e_t1 = dt.datetime.now()
            h, m, s = time_diff(e_t1, start_t)

            logging.info('In epoch:{:03d}|batch:{}, loss:{:4f}, acc:{:4f}, time:{}h{}m{}s'.format(epoch,
                                                                                           step,
                                                                                           loss,
                                                                                           pred.detach(),
                                                                                           h, m, s))

    # 5 保存模型
    #     此处就省略了


def gpu_train(proc_id, n_gpus, GPUS,
              graph_data, gnn_model,
              hidden_dim, n_layers, n_classes, fanouts,
              batch_size=32, num_workers=4, epochs=100, accumulation=1, message_queue=None,
              weights=None, l1_weight=0.0
              output_folder='./output'):
    global writer
    
    device_id = GPUS[proc_id]
    logging.info('Use GPU {} for training ......'.format(device_id))

    # ------------------- 1. Prepare data and split for multiple GPUs ------------------- #
    start_t = dt.datetime.now()
    logging.info('Start graph building at: {}-{} {}:{}:{}'.format(start_t.month,
                                                           start_t.day,
                                                           start_t.hour,
                                                           start_t.minute,
                                                           start_t.second))

    graph, labels, train_nid, val_nid, test_nid, node_feat = graph_data

    train_div, _ = divmod(train_nid.shape[0], n_gpus)
    val_div, _ = divmod(val_nid.shape[0], n_gpus)

    # just use one GPU, give all training/validation index to the one GPU
    if proc_id == (n_gpus - 1):
        train_nid_per_gpu = train_nid[proc_id * train_div: ]
        val_nid_per_gpu = val_nid[proc_id * val_div: ]
    # in case of multiple GPUs, split training/validation index to different GPUs
    else:
        train_nid_per_gpu = train_nid[proc_id * train_div: (proc_id + 1) * train_div]
        val_nid_per_gpu = val_nid[proc_id * val_div: (proc_id + 1) * val_div]
    

    sampler = MultiLayerNeighborSampler(fanouts)
#     print("Before train data loader")
    train_dataloader = NodeDataLoader(graph,
                                      train_nid_per_gpu,
                                      sampler,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=num_workers,
                                      )
    tmp = dt.datetime.now()
    logging.info("Finish train data loader at :{}-{} {}:{}:{}".format(tmp.month,
                                                           tmp.day,
                                                           tmp.hour,
                                                           tmp.minute,
                                                           tmp.second))
    
    val_dataloader = NodeDataLoader(graph,
                                    val_nid_per_gpu,
                                    sampler,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=num_workers,
                                    )
    tmp = dt.datetime.now()
    logging.info("Finish val data loader at :{}-{} {}:{}:{}".format(tmp.month,
                                                           tmp.day,
                                                           tmp.hour,
                                                           tmp.minute,
                                                           tmp.second))

    e_t1 = dt.datetime.now()
    h, m, s = time_diff(e_t1, start_t)
    logging.info('Model built used: {:02d}h {:02d}m {:02}s'.format(h, m, s))

    # ------------------- 2. Build model for multiple GPUs ------------------------------ #
    start_t = dt.datetime.now()
    logging.info('Start Model building at: {}-{} {}:{}:{}'.format(start_t.month,
                                                           start_t.day,
                                                           start_t.hour,
                                                           start_t.minute,
                                                           start_t.second))

    if n_gpus > 1:
        dist_init_method = 'tcp://{}:{}'.format('127.0.0.1', '23456')
        world_size = n_gpus
        dist.init_process_group(backend='nccl',
                                init_method=dist_init_method,
                                world_size=world_size,
                                rank=proc_id)

    in_feat = node_feat.shape[1]
    if gnn_model == 'graphsage':
        model = GraphSageModel(in_feat, hidden_dim, n_layers, n_classes)
    elif gnn_model == 'graphconv':
        model = GraphConvModel(in_feat, hidden_dim, n_layers, n_classes,
                               norm='both', activation=F.relu, dropout=0)
    elif gnn_model == 'graphattn':
        model = GraphAttnModel(in_feat, hidden_dim, n_layers, n_classes,
                               heads=([3] * n_layers), activation=F.relu, feat_drop=0, attn_drop=0)
    else:
        raise NotImplementedError('So far, only support three algorithms: GraphSage, GraphConv, and GraphAttn')

    model = model.to(device_id)

    if n_gpus > 1:
        model = thnn.parallel.DistributedDataParallel(model,
                                                      device_ids=[device_id],
                                                      output_device=device_id)
    e_t1 = dt.datetime.now()
    h, m, s = time_diff(e_t1, start_t)
    logging.info('Model built used: {:02d}h {:02d}m {:02}s'.format(h, m, s))

    # ------------------- 3. Build loss function and optimizer -------------------------- #
    loss_fn = thnn.CrossEntropyLoss(weight=weights).to(device_id)
    optimizer = optim.Adam(model.parameters(), lr=0.004, weight_decay=5e-4)

    earlystoper = early_stopper(patience=2, verbose=False)

    # ------------------- 4. Train model  ----------------------------------------------- #
    logging.info('Plan to train {} epoches \n'.format(epochs))
    
    global_step = 0
    best_val_acc = -1
    
    tmp_loss = []
    tmp_pred = []
    tmp_bs = []
    for epoch in range(epochs):

        # mini-batch for training
        train_loss_list = []
        train_acc_list = []
        model.train()
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            # forward
            batch_inputs, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device_id)
            blocks = [block.to(device_id) for block in blocks]
            # metric and loss
            train_batch_logits = model(blocks, batch_inputs)
            train_loss = loss_fn(train_batch_logits, batch_labels)
            train_loss = train_loss / accumulation
            
            # l1 损失
            if l1_weight > 1e-7:
                train_loss += l1_weight * l1_regularization(model)
            
            tmp_bs.append(batch_labels.shape[0])
            tmp_loss.append(train_loss.cpu().detach().numpy())
            tmp_pred.extend((th.argmax(train_batch_logits, dim=1) == batch_labels).cpu().detach().tolist())
            
            # backward
            train_loss.backward()
            if (step + 1) % accumulation == 0 or (step+1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                
                step_loss = sum([e1 * e2 * accumulation for e1, e2 in zip(tmp_bs, tmp_loss)]) / sum(tmp_bs)
                step_acc = sum(tmp_pred) / len(tmp_pred)
                
                tmp_bs.clear()
                tmp_loss.clear()
                tmp_pred.clear()
                
                train_loss_list.append(step_loss)
                train_acc_list.extend(tmp_pred)
                
                if (step+1) % (accumulation * 10) == 0:
                    logging.info('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, train_acc:{:.4f}'.format(epoch,
                                                                                                    (step+1) // (accumulation),
                                                                                                    step_loss,
                                                                                                    step_acc))
                
                writer.add_scalar("step/loss", step_loss, global_step)
                writer.add_scalar("step/acc", step_acc, global_step)
                global_step += 1       

#             train_loss_list.append(train_loss.cpu().detach().numpy())
#             tr_batch_pred = th.sum(th.argmax(train_batch_logits, dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])
#             train_acc_list.extend((th.argmax(train_batch_logits, dim=1) == batch_labels).cpu().detach().tolist())

#             if step % 10 == 0:
#                 logging.info('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, train_acc:{:.4f}'.format(epoch,
#                                                                                                 step,
#                                                                                                 np.mean(train_loss_list),
#                                                                                                 tr_batch_pred.detach()))
#             writer.add_scalar("step/loss", train_loss.cpu().detach().numpy(), global_step)
#             writer.add_scalar("step/acc", tr_batch_pred.detach(), global_step)
#             global_step += 1
        
        writer.add_scalar("epoch/loss", np.mean(train_loss_list), epoch)
        writer.add_scalar("epoch/acc", np.mean(train_acc_list), epoch)
        

        # mini-batch for validation 每训练完一个epoch就验证一次
        val_loss_list = []
        val_acc_list = []
        model.eval()
        for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
            # forward
            batch_inputs, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device_id)
            blocks = [block.to(device_id) for block in blocks]
            # metric and loss
            val_batch_logits = model(blocks, batch_inputs)
            val_loss = loss_fn(val_batch_logits, batch_labels)

            val_loss_list.append(val_loss.detach().cpu().numpy())
            val_batch_pred = th.sum(th.argmax(val_batch_logits, dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])
            val_acc_list.extend((th.argmax(val_batch_logits, dim=1) == batch_labels).cpu().detach().tolist())
            
            if step % 10 == 0:
                logging.info('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_acc:{:.4f}'.format(epoch,
                                                                                            step,
                                                                                            np.mean(val_loss_list),
                                                                                            val_batch_pred.detach()))
        writer.add_scalar("val_epoch/loss", np.mean(val_loss_list), epoch)
        writer.add_scalar("val_epoch/acc", np.mean(val_acc_list), epoch)
        
        if np.mean(val_acc_list) - best_val_acc >= 0.00001:
            model_path = os.path.join(output_folder, f"model-best-val-acc-{np.mean(val_acc_list):.5}.pth")
            logging.info(f"Saved model with best val acc to {model_path}.\t({best_val_acc} -> {np.mean(val_acc_list)}) .")
            best_val_acc = np.mean(val_acc_list)
            model_para_dict = model.state_dict()
            th.save(model_para_dict, model_path)
            
        
        # put validation results into message queue and aggregate at device 0
        if n_gpus > 1 and message_queue != None:
            message_queue.put(val_loss_list)

            if proc_id == 0:
                for i in range(n_gpus):
                    loss = message_queue.get()
                    print(loss)
                    del loss
        else:
            print(val_loss_list)

    # -------------------------5. Collect stats ------------------------------------#
    # best_preds = earlystoper.val_preds
    # best_logits = earlystoper.val_logits
    #
    # best_precision, best_recall, best_f1 = get_f1_score(val_y.cpu().numpy(), best_preds)
    # best_auc = get_auc_score(val_y.cpu().numpy(), best_logits[:, 1])
    # best_recall_at_99precision = recall_at_perc_precision(val_y.cpu().numpy(), best_logits[:, 1], threshold=0.99)
    # best_recall_at_90precision = recall_at_perc_precision(val_y.cpu().numpy(), best_logits[:, 1], threshold=0.9)

    # plot_roc(val_y.cpu().numpy(), best_logits[:, 1])
    # plot_p_r_curve(val_y.cpu().numpy(), best_logits[:, 1])

    # -------------------------6. Save models --------------------------------------#
    model_path = os.path.join(output_folder, 'dgl_model-' + '{:06d}'.format(np.random.randint(100000)) + '.pth')

    if n_gpus > 1:
        if proc_id == 0:
            model_para_dict = model.state_dict()
            th.save(model_para_dict, model_path)
            # after trainning, remember to cleanup and release resouces
            cleanup()
    else:
        model_para_dict = model.state_dict()
        th.save(model_para_dict, model_path)
    
    logging.info(f"model saved to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGL_SamplingTrain')
    parser.add_argument('--data_path', type=str, help="Path of saved processed data files.")
    parser.add_argument('--gnn_model', type=str, choices=['graphsage', 'graphconv', 'graphattn'],
                        required=True, default='graphsage')
    parser.add_argument('--hidden_dim', nargs='+', type=int, required=True)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument("--fanout", type=str, required=True, help="fanout numbers", default='20,20')
    parser.add_argument('--batch_size', type=int, required=True, default=1)
    parser.add_argument('--GPU', nargs='+', type=int, required=True)
    parser.add_argument('--num_workers_per_gpu', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--out_path', type=str, required=True, help="Absolute path for saving model parameters")
    t_year, t_month, t_day = dt.datetime.now().year, dt.datetime.now().month, dt.datetime.now().day
#     parser.add_argument('--log_dir', type=str, default="./log", help="Path to save log file")
    parser.add_argument('--log_name', type=str, default=f"experiment-{t_year}-{t_month}-{t_day}-{np.random.randint(100000)}")
    parser.add_argument('--accumulation', type=int, default=1, help="accumulation gradient")
    parser.add_argument('--class_weights', action='store_true', help="Use class weights or not.")
    parser.add_argument('--user_infer', action='store_true', help="Use infered train nodes or not.")
    parser.add_argument('--l1_weight', type=float, default=0.0, help="Weight of l1 regularization loss.")
    args = parser.parse_args()
    
    # parse arguments
    BASE_PATH = args.data_path
    MODEL_CHOICE = args.gnn_model
    HID_DIM = args.hidden_dim
    N_LAYERS = args.n_layers
    FANOUTS = [int(i) for i in args.fanout.split(',')]
    BATCH_SIZE = args.batch_size
    GPUS = args.GPU
    WORKERS = args.num_workers_per_gpu
    EPOCHS = args.epochs
    OUT_PATH = args.out_path
    ACCUMULATION = args.accumulation
    CLASS_WEIGHTS = args.class_weights
    USE_INFER = args.use_infer
    L1_WEIGHT =args.l1_weight
    
    exp_dir = os.path.join(OUT_PATH, args.log_name)  # 实验输出目录
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    
    set_logging(exp_dir, "train-log") 
    set_summary_writer(exp_dir, "train-tensorboard")
    
    logging.info(' '.join(sys.argv))    
    logging.info(f"Experiments output will be saved in {exp_dir}.")
    
    # output arguments for logging   
    logging.info('Data path: {}'.format(BASE_PATH))
    logging.info('Used algorithm: {}'.format(MODEL_CHOICE))
    logging.info('Hidden dimensions: {}'.format(HID_DIM))
    logging.info('number of hidden layers: {}'.format(N_LAYERS))
    logging.info('Fanout list: {}'.format(FANOUTS))
    logging.info('Batch size: {}'.format(BATCH_SIZE))
    logging.info('GPU list: {}'.format(GPUS))
    logging.info('Number of workers per GPU: {}'.format(WORKERS))
    logging.info('Max number of epochs: {}'.format(EPOCHS))
    logging.info('Accumulation step: {}'.format(ACCUMULATION))
    logging.info('Class weights: {}'.format(CLASS_WEIGHTS))
    logging.info('Use infered train nodes: {}'.format(USE_INFER))
    logging.info("L1 loss weight: {}".format(L1_WEIGHT))
    logging.info('Output path: {}'.format(OUT_PATH))

    # Retrieve preprocessed data and add reverse edge and self-loop
    graph, labels, train_nid, val_nid, test_nid, node_feat = load_dgl_graph(BASE_PATH, use_infer=USE_INFER)
    train_nid, val_nid = train_val_split(labels.numpy(), seed=444)
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    
    # 类被权重
    weights = th.Tensor([2.39668255, 1.01738769, 0.16509785, 0.16509785, 1.46418911,
       1.73149039, 1.49858725, 0.87378317, 1.92708851, 1.89962726,
       1.73400088, 1.27971203, 0.6062837 , 0.1768355 , 2.06805041,
       1.3066888 , 2.02217888, 1.7373482 , 1.91354507, 1.99570861,
       1.91066021, 1.58436238, 2.16675236])

    # call train with CPU, one GPU, or multiple GPUs
    if GPUS[0] < 0:
        cpu_device = th.device('cpu')
        cpu_train(graph_data=(graph, labels, train_nid, val_nid, test_nid, node_feat),
                  gnn_model=MODEL_CHOICE,
                  n_layers=N_LAYERS,
                  hidden_dim=HID_DIM,
                  n_classes=23,
                  fanouts=FANOUTS,
                  batch_size=BATCH_SIZE,
                  num_workers=WORKERS,
                  device=cpu_device,
                  epochs=EPOCHS,
                  out_path=exp_dir)
    else:
        n_gpus = len(GPUS)

        if n_gpus == 1:
            gpu_train(0, n_gpus, GPUS,
                      graph_data=(graph, labels, train_nid, val_nid, test_nid, node_feat),
                      gnn_model=MODEL_CHOICE, hidden_dim=HID_DIM, n_layers=N_LAYERS, n_classes=23,
                      fanouts=FANOUTS, batch_size=BATCH_SIZE, num_workers=WORKERS, epochs=EPOCHS, accumulation=ACCUMULATION,
                      message_queue=None, weights=weights if CLASS_WEIGHTS else None, l1_weight=L1_WEIGHT, output_folder=exp_dir)
        else:
            message_queue = mp.Queue()
            procs = []
            for proc_id in range(n_gpus):
                p = mp.Process(target=gpu_train,
                               args=(proc_id, n_gpus, GPUS,
                                     (graph, labels, train_nid, val_nid, test_nid, node_feat),
                                     MODEL_CHOICE, HID_DIM, N_LAYERS, 23,
                                     FANOUTS, BATCH_SIZE, WORKERS, EPOCHS, ACCUMULATION,
                                     message_queue, weights if CLASS_WEIGHTS else None, L1_WEIGHT, exp_dir))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()