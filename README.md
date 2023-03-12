# Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models (ZSCL)

This is the official implementation of paper "Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models". Our approach Zero-Shot Continual Learning (ZSCL) aims to mitigate forgetting problem existed in the continual learning of large pretrained vision-language models. This repo includes experiments for Multi-domain Task Increamental Learning (MTIL) in `mtil` and Class Incremental Learning in `cil`.

## Multi-domain Task Increamental Learning

Go into the `mtil` directory.

### Dataset Preparation

- Required Datasets: `ImageNet`, `Conceptual_Captions`
- Target Datasets: `Aircraft`, `Caltech101`,`CIFAR10`, `CIFAR100`, `DTD`, `EuroSAT`, `Flowers`, `Food`, `MNIST`, `OxfordPet`,`StanfordCars`, `SUN397`

Download required datasets (ImageNet and Conceptual_Captions) and datasets to be learned (e.g., DTD, MNIST, ect.) to your experiment directory. You can refer to `datasets.md` for more details.

### Replicate our results

If you want to replicate our results in our paper, you can directly run the command line for that table. For example, if you want to replicate the results of ZSCL, then run `bash scripts/ZSCL.sh`.

More commands are listed in `scripts` directory.

To calculate Transfer, Avg., and Last scores, you can use `results.ipynb`. The results in the paper are also listed in `results.ipynb`.

### Training and Evaluation

Following cammand lines are examples of training and evaluating the model.

```sh
# train from clip model
python -m src.main \
    --train-mode=whole \
    --train-dataset=DTD \
    --lr=1e-5\
    --ls 0.2 \
    --iterations 1000 \
    --method ZSCL \
    --image_loss \
    --text_loss \
    --we \
    --avg_freq 100 \
    --l2 1 \
    --ref-dataset ImageNet \
    --ref-sentences conceptual_captions \
    --save ckpt/exp_000

# evaluation on all datasets
python -m src.main --eval-only \
    --train-mode=whole \
    --eval-datasets=Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397 \
    --load ckpt/exp_000/Flowers.pth

# continual training
python -m src.main \
    --train-mode=whole \
    --train-dataset=MNIST \
    --lr=5e-5 \
    --ls 0.2 \
    --method ZSCL \
    --image_loss \
    --text_loss \
    --we \
    --avg_freq 100 \
    --l2 1 \
    --ref-dataset ImageNet \
    --ref-sentences conceptual_captions \
    --iterations 1000 \
    --save ckpt/exp_000 \
    --load ckpt/exp_000/DTD.pth
```

### Hyperparameters

The meaning of hyperparameters in the command line is as follows:

| params            | name                                        |
| ----------------- | ------------------------------------------- |
| --batch-size      | batch size                                  |
| --iterations      | iterations during training                  |
| --lr              | learning rate                               |
| --method          | approaches for continual learning           |
| --train-model     | components of the model to be trained       |
| --ls              | label smoothing                             |
| --image_loss      | initialize loss for image encoder           |
| --text_loss       | initialize loss for text encoder            |
| --we              | initialize weight ensemble                  |
| --avg_freq        | frequency to conduct weight ensemble        |
| --l2              | initialize weight constraint                |
| --ref-dataset     | dataset for image encoder distillation      |
| --ref-sentences   | texts for text encoder distillation         |
| --dataset_order   | the order of datasets for iCaRL             |

## Class Incremental Learning

### Dataest Preparation

- Required Datasets: `ImageNet`, `Conceptual_Captions`
- Target Datasets: `ImageNet`, `tinyImageNet`, `CIFAR100`

Download required datasets (ImageNet and Conceptual_Captions) and datasets to be learned (e.g., tinyImageNet, CIFAR100 ect.) to your experiment directory.

### Training

First, change your configs in `.yaml` files in configs/class

Then, use below cammand lines to train and evaluate the model. if you want to replicate our results in our paper, you can directly run the command line for that table and run the command line below.

```sh
python main.py \
    --config-path configs/class \
    --config-name imagenet100_10-10-ZSCL.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/imagenet100.yaml"
```

### Hyperparameters

The meaning of hyperparameters in the command line is as follows:

| params              | name                                        |
| -----------------   | ------------------------------------------- |
| --batch-size        | batch size                                  |
| --initial_increment | initial increment                           |
| --lr                | learning rate                               |
| --dataset           | the dataset to be learned                   |
| --method            | appraoch for Continual Learning             |
| --ls                | label smoothing                             |
| --weight_decay      | weight decay                                |
| --text_loss         | initialize loss for text encoder            |
| --we                | initialize weight ensemble                  |
| --avg_freq          | frequency to conduct weight ensemble        |
| --l2                | initialize weight constraint                |
| --ref-dataset       | dataset for image encoder distillation      |
| --ref-sentences     | texts for text encoder distillation         |
| --ce_method         | an optional method to add external texts    |

## Citation

```bibtex
TBD
```

## Acknowledgement

`mil` is built on [wise-ft](https://github.com/mlfoundations/wise-ft), and `cil` is built on [Continual-CLIP](https://github.com/vgthengane/Continual-CLIP). We thank the authors for sharing their codes.
