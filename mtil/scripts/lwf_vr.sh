#!/bin/bash
set -v
set -e
set -x
# app_003_fair
exp_no=lwf_vr_fair 
GPU=6,7
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
lr=(5e-5 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 5e-5 1e-5 1e-5 1e-5 1e-5)

# first dataset
CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
    --train-mode=whole \
    --train-dataset=${dataset[0]} \
    --lr=${lr[0]} \
    --ls 0.2 \
    --iterations 1000 \
    --method ZSCL \
    --image_loss \
    --text_loss \
    --ref-dataset ${dataset[0]} \
    --fair \
    --ref-sentences random \
    --save ckpt/exp_${exp_no}

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
        --ref-dataset ${dataset_cur} \
        --ref-sentences random \
        --ref-model ckpt/exp_${exp_no}/${dataset_pre}.pth \
        --iterations 1000 \
        --fair \
        --save ckpt/exp_${exp_no} \
        --load ckpt/exp_${exp_no}/${dataset_pre}.pth
done
