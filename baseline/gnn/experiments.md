# 2021-10-14 
- cmd : `python3.8 model_train.py --data_path ../../dataset --gnn_model graphsage --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --GPU 0 1 --out_path ./output`
- 单机单卡训练，但是卡在train_dataloader创建那一行
- 生成的模型：dgl_model-057137.pth
- 提交结果：dgl_model-057137-1634652489.csv
- 得分：0.0337646200804949
- 备注：
  1. 得分太低，与训练过程不符，可能是推断时的代码出现问题

# 2021-10-19
- cmd：`python3.8 model_train.py --data_path ../../dataset --gnn_model graphsage --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --GPU 0 1 --out_path ./output`
- 单机单卡训练
- 生成的模型：dgl_model-050209.pth
- 提交结果：dgl_model-050209-1634700087.csv
- 得分：47.3429494971323
- 备注：
    1. 之前出错的原因：`test_logits.argmin(axis=1)`，应该为`test_logits.argmax(axis=1)`