#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=4 \
    eval.py                    \
    --stat-path ../../stats/cifar10/cifar10.test.npz \
	--resume ../results-train/040-DiT-N-2-debug/checkpoints/final.pth \
	--model DiT-N/2 \
	--eval-batch-size 128 \
	--cfg \
	--trainsetup GET \
	--depth 12 \
	--hidden-size 384 \
	--num-heads 12 \
	--cond;
