#!/bin/bash

for i in {1..10}
do
    ((seed=$i + 983))
    python experiments/signal-game/train.py \
        --mode sumsample \
        --lr 0.005 \
        --entropy_coeff 0.01 \
        --batch_size 64 \
        --n_epochs 200 \
        --game_size 16 \
        --latent_size 256 \
        --embedding_size 256 \
        --hidden_size 512 \
        --weight_decay 0. \
        --random_seed ${seed}
done
