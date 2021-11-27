import argparse
import copy
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import dgl
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from model import MLP, MLPLinear, GAT, CorrectAndSmooth
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def evaluate(y_pred, y_true, idx, evaluator):
    return evaluator.eval({
        'y_true': y_true[idx],
        'y_pred': y_pred[idx]
    })['acc']


def main():
    # check cuda
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    # load data
    dataset = DglNodePropPredDataset(name=args.dataset)
    evaluator = Evaluator(name=args.dataset)

    split_idx = dataset.get_idx_split()
    g, labels = dataset[0] # graph: DGLGraph object, label: torch tensor of shape (num_nodes, num_tasks)
#     g = dgl.to_bidirected(g, copy_ndata=True)  # g 是多重图，不支持双向
    g = dgl.add_self_loop(g)
    
    if args.dataset == 'ogbn-arxiv':
        if args.model == 'gat':
            g = dgl.add_reverse_edges(g, copy_ndata=True)
            g = g.add_self_loop()
        else:
            g = dgl.to_bidirected(g, copy_ndata=True)
        
        feat = g.ndata['feat']
        feat = (feat - feat.mean(0)) / feat.std(0)
        g.ndata['feat'] = feat

#     g = g.to(device)
    feats = g.ndata['feat']
    labels = labels.to(device)

    # load masks for train / validation / test
    train_idx = split_idx["train"]#.to(device)
    valid_idx = split_idx["valid"]#.to(device)
    test_idx = split_idx["test"]#.to(device)

    n_features = feats.size()[-1]
    n_classes = dataset.num_classes
    
    # load model
    if args.model == 'mlp':
        model = MLP(n_features, args.hid_dim, n_classes, args.num_layers, args.dropout)
    elif args.model == 'linear':
        model = MLPLinear(n_features, n_classes)
    elif args.model == 'gat':
        model = GAT(in_feats=n_features,
                    n_classes=n_classes,
                    n_hidden=args.hid_dim,
                    n_layers=args.num_layers,
                    n_heads=args.n_heads,
                    activation=F.relu,
                    dropout=args.dropout,
                    attn_drop=args.attn_drop)
    else:
        raise NotImplementedError(f'Model {args.model} is not supported.')
    
    model = model.to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    print(model)
    
    sampler = MultiLayerNeighborSampler(args.fanouts)
    train_dataloader = NodeDataLoader(g,
                                      train_idx,
                                      sampler,
                                      batch_size=args.batchsize,
                                      shuffle=True,
                                      drop_last=False,
#                                       num_workers=4,
                                      )    
    val_dataloader = NodeDataLoader(g,
                                    valid_idx,
                                    sampler,
                                    batch_size=args.batchsize,
                                    shuffle=True,
                                    drop_last=False,
#                                     num_workers=4,
                                    )
    test_dataloader = NodeDataLoader(g,
                                    test_idx,
                                    sampler,
                                    batch_size=args.batchsize,
                                    shuffle=True,
                                    drop_last=False,
#                                     num_workers=4,
                                    )

    if args.pretrain:
        print('---------- Before ----------')
        model.load_state_dict(torch.load(f'base/{args.dataset}-{args.model}.pt'))
        model.eval()

        if args.model == 'gat':
            y_soft = model(g, feats).exp()
        else:
            y_soft = model(feats).exp()

        y_pred = y_soft.argmax(dim=-1, keepdim=True)
        valid_acc = evaluate(y_pred, labels, valid_idx, evaluator)
        test_acc = evaluate(y_pred, labels, test_idx, evaluator)
        print(f'Valid acc: {valid_acc:.4f} | Test acc: {test_acc:.4f}')

        print('---------- Correct & Smoothing ----------')
        cs = CorrectAndSmooth(num_correction_layers=args.num_correction_layers,
                              correction_alpha=args.correction_alpha,
                              correction_adj=args.correction_adj,
                              num_smoothing_layers=args.num_smoothing_layers,
                              smoothing_alpha=args.smoothing_alpha,
                              smoothing_adj=args.smoothing_adj,
                              scale=args.scale)
        
        mask_idx = torch.cat([train_idx, valid_idx])
        if args.model != 'gat':
            y_soft = cs.correct(g, y_soft, labels[mask_idx], mask_idx)
        y_soft = cs.smooth(g, y_soft, labels[mask_idx], mask_idx)
        y_pred = y_soft.argmax(dim=-1, keepdim=True)
        valid_acc = evaluate(y_pred, labels, valid_idx, evaluator)
        test_acc = evaluate(y_pred, labels, test_idx, evaluator)
        print(f'Valid acc: {valid_acc:.4f} | Test acc: {test_acc:.4f}')
    else:
        if args.model == 'gat':
            opt = optim.RMSprop(model.parameters(), lr=args.lr)
        else:
            opt = optim.Adam(model.parameters(), lr=args.lr)

        best_acc = 0
        best_model = copy.deepcopy(model)

        # training
        print('---------- Training ----------')
        for epoch in range(args.epochs):
            print(f"\n{'='*35} Epoch {epoch} {'='*35}")
            if args.model == 'gat':
                adjust_learning_rate(opt, args.lr, epoch)
            
            # mini-batch 训练模型
            model.train()
            train_count = 0
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                blocks = [block.to(device) for block in blocks]
                batch_feats = feats[input_nodes]
                batch_feats = batch_feats.to(device)
#                 print(f"batch_feats: {batch_feats.shape}\tinput_nodes: {input_nodes.shape}\tseeds: {seeds.shape}\tblocks: {[block.num_nodes() for block in blocks]}")
                if args.model == 'gat':
                    logits = model(blocks, batch_feats)
                else:
                    logits = model(batch_feats)
                    
                train_loss = F.nll_loss(logits, labels.squeeze(1)[seeds])
                
                train_count += sum(logits.argmax(axis=1) == labels.squeeze(1)[seeds])
                
                print(('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, train_acc:{:.4f}'.format(epoch,
                                                                                            step,
                                                                                            train_loss.item(),
                                                                                            train_count / seeds.shape[0])))
                opt.zero_grad()
                train_loss.backward()
                opt.step()
                
            # mini-batch 评估模型
            model.eval()
            val_count = 0
            for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                blocks = [block.to(device) for block in blocks]
                batch_feats = feats[input_nodes]
                batch_feats = batch_feats.to(device)
                if args.model == 'gat':
                    logits = model(g, batch_feats)
                else:
                    logits = model(batch_feats)
                    
                val_loss = F.nll_loss(logits, labels.squeeze(1)[seeds])
                
                y_pred = logits.argmax(dim=-1, keepdim=True)
                val_count += sum(y_pred.argmax(axis=1) == labels.squeeze(1)[seeds])
                print(('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_acc:{:.4f}'.format(epoch,
                                                                                            step,
                                                                                            val_loss.item(),
                                                                                            val_count / seeds.shape[0])))

            train_acc = train_count / train_idx.shape[0]
            valid_acc = val_count / train_idx.shape[0]  # evaluate(y_pred, labels, valid_idx, evaluator)

            print(f'Epoch {epoch} | Train acc: {train_acc:.4f} | Valid acc {valid_acc:.4f}')

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model = copy.deepcopy(model)
        
        # testing & saving model
        print('---------- Testing ----------')
        best_model.eval()
        test_count = 0
        for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
            blocks = [block.to(device) for block in blocks]
            batch_feats = feats[input_nodes]
            batch_feats = batch_feats.to(device)
            if args.model == 'gat':
                logits = best_model(blocks, batch_feats)
            else:
                logits = best_model(batch_feats)
                
            test_loss = F.nll_loss(logits, labels.squeeze(1)[seeds])
            
            test_count += sum(logits.argmax(axis=1) == labels.squeeze(1)[seeds])
            
            print(('In epoch:{:03d}|batch:{:04d}, test_loss:{:4f}, test_acc:{:.4f}'.format(epoch,
                                                                                            step,
                                                                                            test_loss.item(),
                                                                                            test_count / seeds.shape[0])))
        
        
        test_acc = test_count / test_idx.shape[0]  # evaluate(y_pred, labels, test_idx, evaluator)
        print(f'Test acc: {test_acc:.4f}')

        if not os.path.exists('base'):
            os.makedirs('base')

        torch.save(best_model.state_dict(), f'base/{args.dataset}-{args.model}.pt')


if __name__ == '__main__':
    """
    Correct & Smoothing Hyperparameters
    """
    parser = argparse.ArgumentParser(description='Base predictor(C&S)')

    # Dataset
    parser.add_argument('--gpu', type=int, default=0, help='-1 for cpu')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', choices=['ogbn-arxiv', 'ogbn-products'])
    # Base predictor
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'linear', 'gat'])
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--hid-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--fanouts', type=int, nargs='+', required=True)
    parser.add_argument('--batchsize', type=int, default=4096)
    # extra options for gat
    parser.add_argument('--n-heads', type=int, default=3)
    parser.add_argument('--attn_drop', type=float, default=0.05)
    # C & S
    parser.add_argument('--pretrain', action='store_true', help='Whether to perform C & S')
    parser.add_argument('--num-correction-layers', type=int, default=50)
    parser.add_argument('--correction-alpha', type=float, default=0.979)
    parser.add_argument('--correction-adj', type=str, default='DAD')
    parser.add_argument('--num-smoothing-layers', type=int, default=50)
    parser.add_argument('--smoothing-alpha', type=float, default=0.756)
    parser.add_argument('--smoothing-adj', type=str, default='DAD')
    parser.add_argument('--scale', type=float, default=20.)

    args = parser.parse_args()
    print(args)

    main()
