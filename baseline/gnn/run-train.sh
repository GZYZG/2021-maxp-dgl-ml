nohup python3.8 model_train.py --data_path ../../dataset \
	--gnn_model graphsage \
	--hidden_dim 192 64 \
	--n_layers 3 \
	--fanout 15,15,15 \
	--batch_size 2048 \
	--GPU 1 \
	--epochs 150 \
	--out_path ./output \
	--num_workers_per_gpu 1\
	--accumulation 4 > nohup.log 2>&1 &
