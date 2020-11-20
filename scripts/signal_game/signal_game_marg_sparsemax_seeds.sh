#!/bin/bash

for i in {1..10}
do
    ((seed=$i + 321))
    python experiments/signal-game/train.py \
        --mode marg \
        --lr 0.005 \
        --entropy_coeff 0.1 \
        --batch_size 64 \
        --n_epochs 500 \
        --game_size 16 \
        --latent_size 256 \
        --embedding_size 256 \
        --hidden_size 512 \
        --normalizer sparsemax \
        --weight_decay 0. \
        --random_seed ${seed}
done
