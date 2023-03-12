#!/bin/bash
set -v
set -e
set -x

exp_no=app_005
GPU=0,1,2,3
dataset=(Flowers SUN397 Aircraft Caltech101 DTD EuroSAT CIFAR100)
lr=(1e-5 1e-5 5e-5 1e-5 1e-5 1e-5 1e-5)

# first dataset
# CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
#     --train-mode=whole \
#     --train-dataset=${dataset[0]} \
#     --lr=${lr[0]} \
#     --ls 0.2 \
#     --iterations 1000 \
#     --method ZSCL \
#     --image_loss \
#     --text_loss \
#     --swa \
#     --avg_freq 100 \
#     --ref-dataset ImageNet \
#     --ref-sentences conceptual_captions \
#     --save ckpt/exp_${exp_no}

for ((i = 1; i < ${#dataset[@]}; i++)); do
    dataset_cur=${dataset[i]}
    dataset_pre=${dataset[i - 1]}

    # continue training
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
        --train-mode=whole \
        --train-dataset=${dataset_cur} \
        --lr=${lr[i]} \
        --ls 0.2 \
        --method ZSCL \
        --image_loss \
        --text_loss \
        --swa \
        --avg_freq 100 \
        --ref-dataset ImageNet \
        --ref-sentences conceptual_captions \
        --iterations 1000 \
        --save ckpt/exp_${exp_no} \
        --load ckpt/exp_${exp_no}/${dataset_pre}.pth
done
