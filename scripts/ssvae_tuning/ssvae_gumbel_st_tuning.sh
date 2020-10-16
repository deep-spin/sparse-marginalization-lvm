#!/bin/bash

for lr in 5e-5 1e-4 5e-4 1e-3 5e-3
do
    for decay in 1e-5 1e-4
    do
        for temp_update in 500 1000
        do
            python experiments/semi_supervised-vae/train.py \
                --mode gs \
                --lr $lr \
                --batch_size 64 \
                --n_epochs 200 \
                --latent_size 10 \
                --temperature_decay $decay \
                --temperature_update_freq $temp_update \
                --straight_through \
                --warm_start_path checkpoints/ssvae/warm_start/softmax/lr-0.001_baseline-runavg/version_0/checkpoints/epoch\=91.ckpt
done
done
done
