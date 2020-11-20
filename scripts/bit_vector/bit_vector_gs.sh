#!/bin/bash

python experiments/bit_vector-vae/train.py \
    --mode gs \
    --lr 0.001 \
    --batch_size 64 \
    --n_epochs 100 \
    --latent_size 32 \
    --weight_decay 0. \
    --temperature_decay 1e-5 \
    --temperature_update_freq 1000 \
    --random_seed 42
