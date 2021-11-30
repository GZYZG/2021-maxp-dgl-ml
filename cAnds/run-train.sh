nohup python3.8 main.py --model gat \
	--gpu 0 \
	--fanouts 12 12 12 \
	--batchsize 1024 \
	--dataset ogbn-arxiv \
	--epochs 50 \
	--lr 0.002 \
	--dropout 0.75 > train.log 2>&1 &
