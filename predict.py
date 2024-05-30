import os
from copy import deepcopy
import numpy as np
import pickle
import argparse
from utils import parse_config, mkdir, rmdir, build_transform
import importlib
import jsonlines

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import Dataset

# from models import EfficientNetV2_s

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions')
    parser.add_argument('-c', '--config', required=True, help='./configs/config.yaml')
    args = parser.parse_args()
    return args

def main(config):
    models = importlib.import_module('models')
    Model = getattr(models, config['Model']['class_name'])

    results_path = os.path.join(config['Dataset']['dataset_path'], 'results')

    rmdir(results_path)
    mkdir(results_path)

    config = parse_config(config_path)

    targets = []
    predictions = []

    for fold_idx in range(config['Dataset']['num_folds']):
        model_path = os.path.join(f"fold_{fold_idx}", "best_bacc.pth")
        
        model = Model()
        model = torch.load(model_path).cuda()
     
        dataset_path = os.path.join(config['Dataset']['dataset_path'], "folds", f"test_{fold_idx}.jsonl")

        test_transform = build_transform(config["Dataset"]["transform"], train=False)

        batch_size = config['Training']['batch_size']
        num_workers = config['Training']['num_workers']

        test_dataset = Dataset(dataset_path, config["Dataset"]["names"], transform=test_transform, return_path=True)
        test_dataloader = DataLoader(test_dataset,  
                                    batch_size=batch_size, 
                                    num_workers=num_workers,
                                    sampler=None)

        num_classes = len(config["Dataset"]["names"])

        images_paths = []

        with jsonlines.open(dataset_path) as reader:
            for sample in reader:
                images_paths.append(os.path.join(config['Dataset']['dataset_path'], sample["image"]))

        for i, batch in enumerate(tqdm(test_dataloader, desc=f"Make predictions for {fold_idx} fold")):
            images, labels, paths = batch
            images = images.cuda()

            for path, label in zip(paths, labels):
                target_sample = {"label":label.item(), "image": path}
                targets.append(deepcopy(target_sample))

            prediction = model(images)
            prediction = nn.Softmax(dim=-1)(prediction).detach().cpu().numpy()

            predictions.append(prediction)
        
    predictions = np.vstack(predictions)

    with open(os.path.join(results_path, 'labels.pickle'), 'wb') as handle:
        pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(results_path, 'predictions.pickle'), 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rmdir(os.path.join(config['Dataset']['dataset_path'], 'folds'))


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    config = parse_config(config_path)

    main(config)











            


