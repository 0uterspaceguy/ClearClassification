import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_warmup as warmup
import argparse

from dataset import Dataset

import os
from os.path import join as pj

import yaml
from yaml import Loader
import numpy as np
import importlib

from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions')
    parser.add_argument('-c', '--config', required=True, help='./configs/config.yaml')
    args = parser.parse_args()
    return args


def main(config: dict):
    for fold_idx in range(config["Dataset"]["num_folds"]):
        exp_label = f"fold_{fold_idx}"

        rmdir(exp_label)
        mkdir(exp_label)

        models = importlib.import_module('models')
        Model = getattr(models, config['Model']['class_name'])
        
        train_path = os.path.join(config['Dataset']['dataset_path'], "folds", f"train_{fold_idx}.jsonl")
        test_path = os.path.join(config['Dataset']['dataset_path'], "folds", f"test_{fold_idx}.jsonl")

        train_transform = build_transform(config["Dataset"]["transform"], train=True)
        test_transform = build_transform(config["Dataset"]["transform"], train=False)

        train_dataset = Dataset(train_path, config['Dataset']["names"], transform=train_transform)
        test_dataset = Dataset(test_path, config['Dataset']["names"], transform=test_transform)
        
        if config['Training']['weighted_sampler']:
            class_counts = [0] * config['Model']['num_classes']
            labels = [train_dataset.label2idx[label] for label in train_dataset.labels] 

            for label in labels:
                class_counts[label] += 1

            num_samples = sum(class_counts)

            class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
            weights = [class_weights[labels[i]] for i in range(int(num_samples))]
            sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
        else:
            sampler=None

        train_dataloader = DataLoader(train_dataset,
                                        batch_size=config['Training']['batch_size'],
                                        num_workers=config['Training']['num_workers'],
                                        sampler=sampler)

        test_dataloader = DataLoader(test_dataset,
                                        batch_size=config['Training']['batch_size'],
                                        num_workers=config['Training']['num_workers'],
                                        sampler=None)
        

        criterion = nn.CrossEntropyLoss().cuda()

        model = Model(num_classes=config["Model"]["num_classes"]).cuda()

        optimizer = optim.Adam(model.parameters(), lr=config['Training']['lr'])

        warmup_period = int(config['Training']['warmup']*len(train_dataloader))
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=warmup_period)
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                    'min', 
                                                                    min_lr=config['Training']['min_lr'],
                                                                    factor=config['Training']['factor'],
                                                                    patience=config['Training']['patience'],)

        best_loss = float('inf')
        best_acc = 0
        best_bacc = 0
        best_f1 = 0

        for epoch in range(config['Training']['epochs']):   
                        
            model, train_loss = train(model, 
                                        optimizer, 
                                        criterion, 
                                        train_dataloader,
                                        warmup_scheduler,
                                        plateau_scheduler,
                                        warmup_period,
                                        epoch)

            test_loss, targets, predictions = test(model, 
                                                    criterion, 
                                                    test_dataloader)


            metrics = count_metrics(targets, predictions)

            metrics.update({
                "Test_loss": test_loss,
                "Train_loss": train_loss,
                "lr": optimizer.param_groups[0]['lr'],
            })

            test_accuracy = metrics['Accuracy']
            test_ballanced_accuracy = metrics['Balanced_accuracy']
            test_f1 = metrics['F1-weighted']

            torch.save(model, pj(exp_label, 'last.pth'))

            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model, pj(exp_label,'best_loss.pth'))
            
            if test_accuracy > best_acc:
                best_acc = test_accuracy
                torch.save(model, pj(exp_label, 'best_acc.pth'))
            
            if test_ballanced_accuracy > best_bacc:
                best_bacc = test_ballanced_accuracy
                torch.save(model, pj(exp_label, 'best_bacc.pth'))

            if test_f1 > best_f1:
                best_f1 = test_f1
                torch.save(model, pj(exp_label, 'best_f1.pth'))
            
        [print(k,v) for k,v in metrics.items()]


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    config = parse_config(config_path)

    main(config)

    





