#!/bin/bash

python experiments/bit_vector-vae/train.py \
    --mode nvil \
    --lr 0.0005 \
    --batch_size 64 \
    --n_epochs 100 \
    --latent_size 128 \
    --weight_decay 0.
