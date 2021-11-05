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
    
---

# 2021-10-24
- cmd：`python3.8 -u model_train.py --data_path ../../dataset --gnn_model graphconv --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --GPU 0 --out_path ./output`
- 单机单卡训练
- 提交结果：
    - 1
        - 模型：model-best-val-acc-0.525.pth
        - 文件：model-best-val-acc-0.525-1635087338.csv
        - 得分：47.20384
    - 2
        - 模型：model-best-val-acc-0.53.pth
        - 文件：model-best-val-acc-0.53-1635170325.csv
        - 得分：47.51357
    - 3 
        - 模型：model-best-val-acc-0.532.pth
        - 文件：model-best-val-acc-0.532-1635125823.csv
        - 得分：47.71841
    - 4 
        - 模型：model-best-val-acc-0.536.pth
        - 文件：model-best-val-acc-0.536-1635125628.csv
        - 得分：47.66349 
        
- 备注
    1. 模型精度提升较慢，但是训练过程更平稳，最终达到了更高的准确度
    2. 出现了过拟合
    
---

# 2021-10-25
- cmd：`python3.8 -u model_train.py --data_path ../../dataset --gnn_model graphconv --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --GPU 0 --epochs 400 --out_path ./output`
- 单机单卡训练
- 结果保存目录：./output/experiment-2021-10-25-21686
- 提交结果：
    - 1
        - 模型：model-best-val-acc-0.533.pth
        - 文件：model-best-val-acc-0.533-1635300695.csv
        - 得分：47.25201
    - 2
        - 模型：model-best-val-acc-0.536.pth
        - 文件：model-best-val-acc-0.536-1635212941.csv
        - 得分：48.14024
    - 3
        - 模型：model-best-val-acc-0.538.pth
        - 文件：model-best-val-acc-0.538-1635212695.csv
        - 得分：47.79900

- 备注：
    1. 训练过程不稳定
    
---

# 2021-10-25
- cmd：`python3.8 -u model_train.py --data_path ../../dataset --gnn_model graphattn --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4096 --GPU 0 --epochs 400 --out_path ./output`
- 单机单卡训练
- 结果保存目录：./output/experiment-2021-10-25-55761
- 提交结果
    - 1
        - 模型：model-best-val-acc-0.514.pth
        - 文件：model-best-val-acc-0.514-1635301311.csv
        - 得分：46.89095
        
- 备注
    1. 训练过程很不稳定，相交其他两个模型更不稳定
    2. 在训练集和验证集上的表现不如其他两种模型（`graphsage`和`graphconv`）

---

# 2021-10-27
- cmd：`python3.8 -u model_train.py --data_path ../../dataset --gnn_model graphconv --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 8192 --GPU 0 --epochs 400 --out_path ./output`
- 单机单卡训练
- 结果保存目录：./output/experiment-2021-10-27-610
- `batch_size=8192`
- 提交结果
    - 1
        - 模型：model-best-val-acc-0.541.pth
        - 文件：model-best-val-acc-0.541-1635399989.csv
        - 得分：48.11278
    - 2 
        - 模型：model-best-val-acc-0.54.pth
        - 文件：model-best-val-acc-0.54-1635400182.csv
        - 得分：48.09883
    - 3
        - 模型：model-best-val-acc-0.538.pth
        - 文件：model-best-val-acc-0.538-1635400360.csv
        - 得分：48.04165
    - 4 
        - 模型：model-best-val-acc-0.537.pth
        - 文件：model-best-val-acc-0.537-1635504221.csv
        - 得分：48.21813
    - 5
        - 模型：model-best-val-acc-0.536.pth
        - 文件：model-best-val-acc-0.536-1635504407.csv
        - 得分：47.88498
- 备注
    1. 增大了batch_size
    2. 相较于`bs=4096`，准确度确实提高了
---

# 2021-10-29
- cmd：`python3.8 -u model_train.py --data_path ../../dataset --gnn_model graphconv --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 2048 --GPU 0 --epochs 400 --out_path ./output`
- 单机单卡训练
- 结果保存目录：./output/experiment-2021-10-29-22556
- `batch_size=2048`
- 提交结果
    - 1
        - 模型：model-best-val-acc-0.537.pth
        - 文件：model-best-val-acc-0.537-1635564362.csv
        - 得分：47.59326
    - 2 
        - 模型：model-best-val-acc-0.536.pth
        - 文件：model-best-val-acc-0.536-1635567538.csv
        - 得分：47.58696
    - 3
        - 模型：model-best-val-acc-0.534.pth
        - 文件：model-best-val-acc-0.534-1635567785.csv
        - 得分：47.74812
        
--- 

# 2021-10-30
- cmd：`python3.8 -u model_train.py --data_path ../../dataset --gnn_model graphconv --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 8192 --GPU 0 --epochs 300 --out_path ./output`
- 单机单卡
- **对特征进行了标准化**
- 结果保存目录：./output/experiment-2021-10-30-72944
- `batch_size=8192`
- 提交结果
    - 1
        - 模型：model-best-val-acc-0.545.pth
        - 文件：model-best-val-acc-0.545-1635647916.csv
        - 得分：29.12986
    - 2
        - 模型：model-best-val-acc-0.542.pth
        - 文件：model-best-val-acc-0.542-1635648296.csv
        - 得分：28.51850
- 备注
    1. 线下表现很好，但是线上效果很差
    2. 400 epochs已经发生了过拟合现象
    
---

# 2021-10-31
- cmd：`python3.8 -u model_train.py --data_path ../../dataset --gnn_model graphconv --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 8192 --GPU 0 --epochs 300 --out_path ./output`
- 单机单卡
- **对特征进行了归一化**
- 结果保存目录：./output/experiment-2021-10-31-24989
- `batch_size=8192`
- 提交结果
    - 1
        - 模型：model-best-val-acc-0.425.pth
        - 文件：model-best-val-acc-0.545-1635647916.csv
        - 得分：25.71198
- 备注
    1. 注意改了.py文件后，在jupyter中要restart kernel才会生效！！！
    
--- 

# 2021-10-31
- cmd：`python3.8 -u model_train.py --data_path ../../dataset --gnn_model graphconv --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 8192 --GPU 0 --epochs 300 --out_path ./output`
- 单机单卡
- **对特征进行了标准化**
- 结果保存目录：./output/experiment-2021-10-31-42423
- `batch_size=8192`
- 提交结果
    - 1
        - 模型：model-best-val-acc-0.547.pth
        - 文件：model-best-val-acc-0.547-1635739515.csv
        - 得分：48.46484
    - 2
        - 模型：model-best-val-acc-0.545.pth
        - 文件：model-best-val-acc-0.545-1635827255.csv
        - 得分：48.30141
    - 3
        - 模型：model-best-val-acc-0.543.pth
        - 文件：model-best-val-acc-0.543-1635827448.csv
        - 得分：48.37300
    - 4 
        - 模型：model-best-val-acc-0.539.pth
        - 文件：model-best-val-acc-0.539-1635827596.csv
        - 得分：48.29151
    - 5
        - 模型：model-best-val-acc-0.547.pth
        - 文件：model-best-val-acc-0.547-adjusted-1635849382.csv
        - 得分：47.80485
- 备注
    1. 使用了CAN对分类概率进行后处理，但是没有提升，反而下降了一点
    
--- 

# 2021-11-03
- cmd：`python3.8 -u model_train.py --data_path ../../dataset --gnn_model graphconv --hidden_dim 128 --n_layers 3 --fanout 10,10,10 --batch_size 512 --GPU 0 --epochs 200 --out_path ./output --num_workers_per_gpu 1`
- 结果保存目录：./output/experiment-2021-11-3-14969
- 标准化后的特征
- 单机单卡
- `batch_size=512 fanout=10,10,10 hidden_dim=128 n_layers=3`
- 提交结果
    - 1
        - 模型：model-best-val-acc-0.551.pth
        - 文件：model-best-val-acc-0.551-1635943575.csv
        - 得分：48.14430
    - 2
        - 模型：model-best-val-acc-0.554.pth
        - 文件：model-best-val-acc-0.554-1636035845.csv
        - 得分：48.31582
    - 3
        - 模型：model-best-val-acc-0.553.pth
        - 文件：model-best-val-acc-0.553-1636036129.csv
        - 得分：48.13799
- 备注
    1. 增加层数后，对显存要求大大提高
    2. 验证集上效果达到了最佳，但是线上不佳，可能是因为batch_size的原因
    3. 就目前的情况来看不需要200 epochs
