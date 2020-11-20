#!/bin/bash

for i in {1..10}
do
    ((seed=$i + 321))
    python experiments/signal-game/train.py \
        --mode sfe \
        --lr 0.001 \
        --entropy_coeff 0.05 \
        --batch_size 64 \
        --n_epochs 500 \
        --game_size 16 \
        --latent_size 256 \
        --embedding_size 256 \
        --hidden_size 512 \
        --loss_type acc \
        --baseline_type sample \
        --weight_decay 0. \
        --random_seed ${seed}
done
