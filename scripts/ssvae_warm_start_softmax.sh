#!/bin/bash

python  experiments/semi_supervised-vae/train.py \
            --n_epochs 100 \
            --lr 1e-3 \
            --labeled_only \
            --normalizer softmax \
            --batch_size 64

