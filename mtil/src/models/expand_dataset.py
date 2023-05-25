import os
import re

from tqdm import tqdm
import numpy as np

SHIFT = 100

class ExpandedDataset:

    def __init__(self, args, dataset, imagenet):
        self.dataset = dataset
        self.train_dataset = dataset.train_dataset
        self.imagenet = imagenet
        self.imagenet_train_dataset = imagenet.train_dataset
        self.shift = SHIFT # should be num_class of the ImageNetSM
        self.out = []
        self.expand()

    def expand(self):
        print("[Expanding] add ImageNet")
        for i in tqdm(np.arange(3)):
            images = self.imagenet_train_dataset[i]["images"]
            labels = self.imagenet_train_dataset[i]["labels"]
            self.out.append(
                (images, labels)
            )
        print("[Expanding] add target_dataset")
        for j in tqdm(np.arange(3)):
            images = self.train_dataset[j][0]
            labels = self.train_dataset[j][1] + self.shift
            self.out.append(
                (images, labels)
            )

    def get(self):
        return self.out






    