#!/bin/bash

python experiments/bit_vector-vae/train.py \
    --mode sfe \
    --lr 0.001 \
    --batch_size 64 \
    --n_epochs 100 \
    --latent_size 32 \
    --baseline_type sample \
    --weight_decay 0.
