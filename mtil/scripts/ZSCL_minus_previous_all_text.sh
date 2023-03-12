#!/bin/bash
set -v
set -e
set -x

dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
lr=(5e-5 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5)

# first dataset
CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
    --train-mode=whole \
    --train-dataset=Aircraft\
    --lr=5e-5 \
    --ls 0.2 \
    --iterations 1000 \
    --method lwa \
    --image_loss \
    --text_loss \
    --ref-dataset ImageNet \
    --text-datasets Aircraft \
    --save ckpt/exp_021

CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
    --train-mode=whole \
    --train-dataset=Caltech101 \
    --lr=1e-5 \
    --ls 0.2 \
    --method lwa \
    --image_loss \
    --text_loss \
    --ref-dataset ImageNet \
    --text-datasets Aircraft,Caltech101 \
    --iterations 1000 \
    --save ckpt/exp_021 \
    --load ckpt/exp_021/Aircraft.pth

CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
    --train-mode=whole \
    --train-dataset=CIFAR100 \
    --lr=1e-5 \
    --ls 0.2 \
    --method lwa \
    --image_loss \
    --text_loss \
    --ref-dataset ImageNet \
    --text-datasets Aircraft,Caltech101,CIFAR100 \
    --iterations 1000 \
    --save ckpt/exp_021 \
    --load ckpt/exp_021/Caltech101.pth

CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
    --train-mode=whole \
    --train-dataset=DTD \
    --lr=1e-5 \
    --ls 0.2 \
    --method lwa \
    --image_loss \
    --text_loss \
    --ref-dataset ImageNet \
    --text-datasets Aircraft,Caltech101,CIFAR100,DTD \
    --iterations 1000 \
    --save ckpt/exp_021 \
    --load ckpt/exp_021/CIFAR100.pth


CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
    --train-mode=whole \
    --train-dataset=EuroSAT \
    --lr=1e-5 \
    --ls 0.2 \
    --method lwa \
    --image_loss \
    --text_loss \
    --ref-dataset ImageNet \
    --text-datasets Aircraft,Caltech101,CIFAR100,DTD,EuroSAT \
    --iterations 1000 \
    --save ckpt/exp_021 \
    --load ckpt/exp_021/DTD.pth



CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
    --train-mode=whole \
    --train-dataset=Flowers \
    --lr=1e-5 \
    --ls 0.2 \
    --method lwa \
    --image_loss \
    --text_loss \
    --ref-dataset ImageNet \
    --text-datasets Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers \
    --iterations 1000 \
    --save ckpt/exp_021 \
    --load ckpt/exp_021/EuroSAT.pth

CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
    --train-mode=whole \
    --train-dataset=Food \
    --lr=1e-5 \
    --ls 0.2 \
    --method lwa \
    --image_loss \
    --text_loss \
    --ref-dataset ImageNet \
    --text-datasets Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food \
    --iterations 1000 \
    --save ckpt/exp_021 \
    --load ckpt/exp_021/Flowers.pth

CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
    --train-mode=whole \
    --train-dataset=MNIST \
    --lr=1e-5 \
    --ls 0.2 \
    --method lwa \
    --image_loss \
    --text_loss \
    --ref-dataset ImageNet \
    --text-datasets Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST \
    --iterations 1000 \
    --save ckpt/exp_021 \
    --load ckpt/exp_021/Food.pth

CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
    --train-mode=whole \
    --train-dataset=OxfordPet \
    --lr=1e-5 \
    --ls 0.2 \
    --method lwa \
    --image_loss \
    --text_loss \
    --ref-dataset ImageNet \
    --text-datasets Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet \
    --iterations 1000 \
    --save ckpt/exp_021 \
    --load ckpt/exp_021/MNIST.pth

CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
    --train-mode=whole \
    --train-dataset=StanfordCars \
    --lr=1e-5 \
    --ls 0.2 \
    --method lwa \
    --image_loss \
    --text_loss \
    --ref-dataset ImageNet \
    --text-datasets Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars \
    --iterations 1000 \
    --save ckpt/exp_021 \
    --load ckpt/exp_021/OxfordPet.pth


CUDA_VISIBLE_DEVICES=0,1 python -m src.main \
    --train-mode=whole \
    --train-dataset=SUN397 \
    --lr=1e-5 \
    --ls 0.2 \
    --method lwa \
    --image_loss \
    --text_loss \
    --ref-dataset ImageNet \
    --text-datasets Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397 \
    --iterations 1000 \
    --save ckpt/exp_021 \
    --load ckpt/exp_021/StanfordCars.pth


