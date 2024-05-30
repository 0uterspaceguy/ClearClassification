import os
import argparse
from utils import parse_config, mkdir, rmdir, rmfile
from random import shuffle
from sklearn.model_selection import KFold
import yaml
import jsonlines

def parse_args():
    parser = argparse.ArgumentParser(description='Split dataset into folds')
    parser.add_argument('-c', '--config', required=True, help='./configs/config.yaml')
    args = parser.parse_args()
    return args

def main(config_path):
    config = parse_config(config_path)

    num_folds = config['Dataset']['num_folds']

    dataset_path = config['Dataset']['dataset_path']
    annotations_path = os.path.join(dataset_path, 'labels.jsonl')

    all_samples = []
    with jsonlines.open(annotations_path) as reader:
        for sample in reader:
            all_samples.append(sample)

    shuffle(all_samples)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    rmdir(os.path.join(dataset_path, "folds"))
    mkdir(os.path.join(dataset_path, "folds"))


    for fold_idx, (train_index, test_index) in enumerate(kf.split(all_samples)):
        fold_annotations_path = os.path.join(dataset_path, f"fold_{fold_idx}.jsonl")

        train_fold_path = os.path.join(dataset_path, "folds", f"train_{fold_idx}.jsonl")
        test_fold_path = os.path.join(dataset_path, "folds", f"test_{fold_idx}.jsonl")

        rmfile(train_fold_path)
        rmfile(test_fold_path)

        train_samples = [all_samples[train_id] for train_id in train_index]
        test_samples = [all_samples[test_id] for test_id in test_index]

        with jsonlines.open(train_fold_path, 'w') as train_file:
            train_file.write_all(train_samples)

        with jsonlines.open(test_fold_path, 'w') as test_file:
            test_file.write_all(test_samples)
     
if __name__ == "__main__":
    args = parse_args()
    main(args.config)