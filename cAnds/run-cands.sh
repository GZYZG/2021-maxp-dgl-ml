nohup python3.8 main.py --model gat \
    --fanouts 12 12 12 \
	--gpu 0 \
	--dataset ogbn-arxiv \
	--pretrain > cands.log 2>&1 &
