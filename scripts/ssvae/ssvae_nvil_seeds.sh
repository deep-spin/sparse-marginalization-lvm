#!/bin/bash

for i in {1..10}
do
    ((seed=$i + 42))
    python experiments/semi_supervised-vae/train.py \
        --mode nvil \
        --random_seed ${seed} \
        --lr 0.001 \
        --batch_size 64 \
        --n_epochs 200 \
        --latent_size 10 \
        --warm_start_path checkpoints/ssvae/warm_start/softmax/lr-0.001_baseline-runavg/version_0/checkpoints/epoch\=91.ckpt
done
