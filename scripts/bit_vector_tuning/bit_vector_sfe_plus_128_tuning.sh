#!/bin/bash

for lr in 0.0005 0.001 0.002
do
    python experiments/bit_vector-vae/train.py \
        --mode sfe \
        --lr $lr \
        --batch_size 64 \
        --n_epochs 100 \
        --latent_size 128 \
        --baseline_type sample \
        --weight_decay 0.
done
