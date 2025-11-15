#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=4 \
	train.py \
	--cond \
    --data-path  /u/vsundaresha/datasets/EDM-Cond-CIFAR-1M\
    --stat-path /u/vsundaresha/codes/diffusion/stats/cifar10/cifar10.train.npz \
	--model DiT-N/2 \
	--epochs 100 \
	--loss multione \
	--trainsetup MultiOne \
	--sigmatime \
	--depth 6 \
	--hidden-size 128 \
	--num-heads 8 \
