#!/bin/bash

for lr in 5e-5 1e-4 5e-4 1e-3 5e-3
do
    python experiments/bit_vector-vae/train.py \
        --mode sparsemap \
        --lr $lr \
        --batch_size 64 \
        --n_epochs 100 \
        --latent_size 32 \
        --weight_decay 0.
done
