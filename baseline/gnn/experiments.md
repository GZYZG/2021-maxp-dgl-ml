# 2021-10-14 
- cmd : `python3.8 model_train.py --data_path ../../dataset --gnn_model graphsage --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --GPU 0 1 --out_path ./output`
- 单机单卡训练，但是卡在train_dataloader创建那一行
- 生成的模型：dgl_model-057137.pth
- 提交结果：dgl_model-057137-1634652489.csv
- 得分：0.0337646200804949
- 备注：
  1. 得分太低，与训练过程不符，可能是推断时的代码出现问题

---

# 2021-10-19
- cmd：`python3.8 model_train.py --data_path ../../dataset --gnn_model graphsage --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --GPU 0 1 --out_path ./output`
- 单机单卡训练
- 生成的模型：dgl_model-050209.pth
- 提交结果：dgl_model-050209-1634700087.csv
- 得分：47.3429494971323
- 备注：
    1. 之前出错的原因：`test_logits.argmin(axis=1)`，应该为`test_logits.argmax(axis=1)`
    
---

# 2021-10-22
- cmd：`python3.8 model_train.py --data_path ../../dataset --gnn_model graphsage --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --GPU 0 1 --out_path ./output`
- 单机单卡训练
- 生成的模型：dgl_model-026428.pth
- 提交结果：dgl_model-026428-1634912673.csv
- 得分：47.06788
- 备注：
    1. 没有调参，但是最后一个epoch的模型并不是最好的结果
    
--- 

# 2021-10-23
- cmd：`python3.8 model_train.py --data_path ../../dataset --gnn_model graphsage --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --GPU 0 1 --out_path ./output`
- 单机单卡训练
- 生成的模型：dgl_model-094040.pth
- 提交结果：dgl_model-094040-1634963249.csv
- 得分：47.59551
- 备注：
    1. 没有调参，但是最后一个epoch的模型并不是最好的结果
    2. 通过记录acc和损失的变化，应该还处于欠拟合阶段
    
---


# 2021-10-23
- cmd：`python3.8 model_train.py --data_path ../../dataset --gnn_model graphsage --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --epochs 200 --GPU 0 1 --out_path ./output`
- 单机单卡训练
- 生成的模型：dgl_model-021702.pth
- 提交结果：dgl_model-021702-1634976941.csv
- 得分：47.37987
- 备注：
    1. 训练200epochs
    2. 通过观察tensorboard的可视化结果，可以看出训练过程很不稳定，loss和acc震荡的很明显
    3. 并没有使用训练过程中最好的模型参数，使用的最后训练完的模型
    
---

# 2021-10-23
- cmd：`python3.8 model_train.py --data_path ../../dataset --gnn_model graphsage --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --epochs 200 --GPU 0 1 --out_path ./output`
- 单机单卡训练
- 生成的模型：model-best-val-acc-0.526.pth
- 提交结果：
    - 1 
        - 模型：model-best-val-acc-0.523.pth
        - 文件：model-best-val-acc-0-1635056614.csv
        - 得分：47.42128
    - 2
        - 模型：model-best-val-acc-0.524.pth
        - 文件：model-best-val-acc-0-1634991837.csv
        - 得分： 47.71031
    - 3
        - 模型：model-best-val-acc-0.525.pth
        - 文件：model-best-val-acc-0-1635056287.csv
        - 得分：47.68825
    - 4
        - 模型：model-best-val-acc-0.526.pth
        - 文件：model-best-val-acc-0-1634991285.csv 
        - 得分：47.55454   
- 备注：
    1. 通过观察tensorboard的可视化结果，可以看出训练过程很不稳定，loss和acc震荡的很明显
    2. 出现了过拟合
    3. 在第101个epoch保存的模型表现最好，之后开始出现过拟合
    

