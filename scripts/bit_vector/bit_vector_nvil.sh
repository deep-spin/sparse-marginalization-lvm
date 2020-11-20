#!/bin/bash

python experiments/bit_vector-vae/train.py \
    --mode nvil \
    --lr 0.001 \
    --batch_size 64 \
    --n_epochs 100 \
    --latent_size 32 \
    --weight_decay 0.
