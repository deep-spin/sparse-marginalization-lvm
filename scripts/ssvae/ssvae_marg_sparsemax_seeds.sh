#!/bin/bash

for i in {1..10}
do
    ((seed=$i + 42))
    python experiments/semi_supervised-vae/train.py \
        --mode marg \
        --normalizer sparsemax \
        --random_seed ${seed} \
        --lr 0.0005 \
        --batch_size 64 \
        --n_epochs 200 \
        --latent_size 10 \
        --warm_start_path checkpoints/ssvae/warm_start/sparsemax/lr-0.001_baseline-runavg/version_0/checkpoints/epoch\=99.ckpt
done
