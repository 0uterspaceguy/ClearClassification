import yaml 
import os
from shutil import rmtree
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

import torch
from torchvision import transforms

def parse_config(path: str) -> dict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def mkdir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def rmdir(path: str) -> None:
    if os.path.exists(path):
        rmtree(path)

def rmfile(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)

def mean(x) -> float:
    return sum(x)/len(x) if len(x) else 0

def count_metrics(targets, predictions):
    f1 = f1_score(targets, predictions, average='weighted')
    accuracy = accuracy_score(targets, predictions)
    balanced_accuracy = balanced_accuracy_score(targets, predictions)

    metrics = {
        'F1-weighted':f1,
        'Accuracy':accuracy,
        'Balanced_accuracy':balanced_accuracy,
    }
    return metrics

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def test(model, 
          criterion, 
          dataloader):

    model.eval()

    losses = []
    predictions = []
    targets = []

    for i, batch in enumerate(tqdm(dataloader, desc="Validation")):
        images, labels = (t.cuda().float() for t in batch)
        labels = labels.type(torch.LongTensor).cuda()


        output = model(images)

        loss = criterion(output, labels)
        losses.append(loss.item())

        output = output.argmax(dim=-1).float()

        predictions.extend(output.tolist())

        targets.extend(labels.tolist())
    
    return mean(losses), targets, predictions


def train(model,
           optimizer, 
           criterion, 
           dataloader,
           warmup_scheduler,
           lr_scheduler,
           warmup_period,
           epoch):

    model.train()

    mean_loss = []

    for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch: {epoch}")):
        images, labels = (t.cuda().float() for t in batch)
        labels = labels.type(torch.LongTensor).cuda()

        output = model(images)

        optimizer.zero_grad()

        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()

        with warmup_scheduler.dampening():
            if (i + 1 == len(dataloader)) and (warmup_scheduler.last_step + 1 >= warmup_period):
                lr_scheduler.step(mean(mean_loss)) # once for epoch and only after warmup done

        mean_loss.append(loss.detach().cpu().item())

    return model, mean(mean_loss)


def build_transform(transform_config, train=True):
    transform = []

    transform.append(transforms.Resize(**transform_config["resize"]))

    if train:

        if "hflip" in transform_config:
            transform.append(transforms.RandomHorizontalFlip(**transform_config["hflip"]))
        
        if "vflip" in transform_config:
            transform.append(transforms.RandomVerticalFlip(**transform_config["vflip"]))

        if "perspective" in transform_config:
            transform.append(transforms.RandomPerspective(**transform_config["perspective"]))

        if "color_jitter" in transform_config:
            transform.append(transforms.ColorJitter(**transform_config["color_jitter"]))

        if "random_crop" in transform_config:
            transform.append(transforms.RandomCrop(**transform_config["random_crop"]))

        if "blur" in transform_config:
            transform.append(transforms.GaussianBlur(**transform_config["blur"]))

        if "rotation" in transform_config:
            transform.append(transforms.RandomRotation(**transform_config["rotation"]))

    transform.append(transforms.ToTensor())

    return transforms.Compose(transform)
    


            


