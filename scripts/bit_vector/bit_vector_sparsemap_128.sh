#!/bin/bash

python experiments/bit_vector-vae/train.py \
    --mode sparsemap \
    --lr 0.002 \
    --batch_size 16 \
    --n_epochs 100 \
    --latent_size 128 \
    --weight_decay 0. \
    --random_seed 42
