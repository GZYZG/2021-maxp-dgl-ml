nohup python3.8 main.py --model gat \
	--gpu 0 \
	--fanouts 12 12 12 \
	--batchsize 4096 \
	--dataset ogbn-products \
	--epochs 200 \
	--lr 0.002 \
	--dropout 0.75 > train.log 2>&1 &
