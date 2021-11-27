nohup python3.8 main.py --model gat \
	--gpu 1 \
	--dataset ogbn-products \
	--pretrain > cands.log 2>&1 &
