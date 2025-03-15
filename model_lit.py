import collections.abc
import copy
import datetime
import imageio
import importlib
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import PIL
import random
import scipy.stats
import shutil
import skimage
import skimage.filters
import sklearn
import sklearn.base
import sklearn.exceptions
import sklearn.metrics
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import Accuracy
import tqdm
import tqdm.auto
import tqdm.notebook
import warnings
from torchvision.datasets import ImageFolder

'''NUM_CLASSES = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100

TRAIN_DIR = r'/srv/fast1/y.pchelitsev/datasets/chest-x-ray/sakha-tb/Sakha-TB-8bit/Train'
VAL_DIR = r'/srv/fast1/y.pchelitsev/datasets/chest-x-ray/sakha-tb/Sakha-TB-8bit/Val'
TEST_DIR = r'/srv/fast1/y.pchelitsev/datasets/chest-x-ray/sakha-tb/Sakha-TB-8bit/Test'

# Для поиска картинок в папках
IMG_EXT = [x[1:] for x in torchvision.datasets.folder.IMG_EXTENSIONS]# ['bmp', 'jpeg', 'jpg', 'png', 'tif', 'tiff']

# Среднее и стандартное отклонение датасета ImageNet. Нужно потому, что используем предобученные на ImageNet модели.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EPS = 1e-7'''

class DataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, test_dir, batch_size=32):
        super(DataModule, self).__init__()
        
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        
        # Нормализация для ImageNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if not os.path.exists(self.train_dir) or not os.path.exists(self.val_dir):
                raise ValueError("Train or validation directory does not exist.")
            self.train_dataset = ImageFolder(root=self.train_dir, transform=self.transform)
            self.val_dataset = ImageFolder(root=self.val_dir, transform=self.transform)
        
        if stage == 'test' or stage is None:
            if not os.path.exists(self.test_dir):
                raise ValueError("Test directory does not exist.")
            self.test_dataset = ImageFolder(root=self.test_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

class COVIDModel(pl.LightningModule):
    def __init__(self, num_classes, freeze_encoder=True):
        super().__init__()

        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder

        self.encoder = models.efficientnet_v2_m(weights='IMAGENET1K_V1')

        # Заморозка слоев EfficientNetV2-M
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Заменяем последний слой классификатора
        in_features = self.encoder.classifier[1].in_features
        self.encoder.classifier[1] = nn.Linear(in_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        self.scheduler_step_size = 10
        self.scheduler_gamma = 0.1

        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def forward(self, x):
        return self.encoder(x)

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  # форма (batch_size, num_classes)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
    
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    
    def on_train_epoch_end(self):
        # Сохраняем результаты текущей эпохи
        self.train_loss_history.append(self.trainer.callback_metrics['train_loss'].item())
        self.train_acc_history.append(self.trainer.callback_metrics['train_acc'].item())

    def on_validation_epoch_end(self):
        # Сохраняем результаты текущей эпохи
        self.val_loss_history.append(self.trainer.callback_metrics['val_loss'].item())
        self.val_acc_history.append(self.trainer.callback_metrics['val_acc'].item())

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = {
            'scheduler': optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma),
            'interval': 'epoch',
            'frequency': 1,
        }
        return [optimizer], [scheduler]

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=True,
    mode='min'
)

#model.save_to_checkpoint(SAVE_PATH + '/model.ckpt')
