#!/bin/bash

for lr in 0.0005 0.001 0.002
do
    for decay in 1e-5 1e-4
    do
        for temp_update in 500 1000
        do
            python experiments/bit_vector-vae/train.py \
                --mode gs \
                --lr $lr \
                --batch_size 64 \
                --n_epochs 100 \
                --latent_size 32 \
                --weight_decay 0. \
                --temperature_decay $decay \
                --temperature_update_freq $temp_update
done
done
done
