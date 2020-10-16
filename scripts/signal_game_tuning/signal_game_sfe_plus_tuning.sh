#!/bin/bash

for lr in 0.01 0.005 0.001
do
    for entreg in 0.1 0.05 0.01
    do
    python experiments/signal-game/train.py \
        --mode sfe \
        --lr $lr \
        --entropy_coeff $entreg \
        --batch_size 64 \
        --n_epochs 200 \
        --game_size 16 \
        --latent_size 256 \
        --embedding_size 256 \
        --hidden_size 512 \
        --loss_type acc \
        --baseline_type sample \
        --weight_decay 0.
done
done
