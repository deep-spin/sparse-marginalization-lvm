#!/bin/bash

python experiments/bit_vector-vae/train.py \
    --mode topksparse \
    --lr 0.002 \
    --batch_size 64 \
    --n_epochs 100 \
    --latent_size 32 \
    --weight_decay 0. \
    --random_seed 42
