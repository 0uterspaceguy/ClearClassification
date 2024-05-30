import os 
import pickle
import argparse
from utils import parse_config, mkdir, rmdir
import matplotlib
from matplotlib import pyplot as plt
import json
from shutil import copyfile as cp
import numpy as np
import cv2
from copy import deepcopy
import jsonlines

from cleanlab import Datalab
import math

def plot_label_issue_examples(label_issue_indices, 
                                given_labels, 
                                predicted_label,
                                dataset,
                                names, 
                                save_path,):
    ncols = 5
    nrows = 2

    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(1.5 * ncols, 1.5 * nrows))
    axes_list = axes.flatten()

    for i, ax in enumerate(axes_list):
        idx = int(label_issue_indices[i])

        ax.set_title(
            f"id: {idx}\n Given: {names[given_labels[i]]}\n Predicted: {names[predicted_label[i]]}",
            fontdict={"fontsize": 8},
        )


        image = cv2.imread(dataset[idx]["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        ax.imshow(image)
        ax.axis("off")

    plt.subplots_adjust(hspace=0.7)
    plt.savefig(save_path)
    matplotlib.pyplot.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze results')
    parser.add_argument('-c', '--config', required=True, help='./configs/config.yaml')
    args = parser.parse_args()
    return args


def main(config_path):
    config = parse_config(config_path)

    class_names = config['Dataset']['names']
    dataset_path = config['Dataset']['dataset_path']

    images_path = os.path.join(dataset_path, 'images')

    with open(os.path.join(dataset_path, 'results', 'labels.pickle'), 'rb') as handle:
        labels = pickle.load(handle)

    with open(os.path.join(dataset_path, 'results', 'predictions.pickle'), 'rb') as handle:
        predictions = pickle.load(handle)


    lab = Datalab(data=labels, label_name="label") 
    lab.find_issues(pred_probs=predictions)
    lab.report()


    label_issues = lab.get_issues("label")
    label_issues_df = label_issues.query("is_label_issue").sort_values("label_score")

    label_issue_indices = label_issues_df.index.values
    given_labels = label_issues_df.given_label.values
    predicted_labels = label_issues_df.predicted_label.values

    if config['do_visualize']:

        for border in range(min(len(label_issues_df), config['num_images_to_vis'])//10):
            indeces_batch = label_issue_indices[border*10: (border+1)*10]
            given_labels_batch = given_labels[border*10: (border+1)*10]
            predicted_labels_batch = predicted_labels[border*10: (border+1)*10]

            plot_label_issue_examples(indeces_batch, 
                                        given_labels_batch, 
                                        predicted_labels_batch,
                                        labels,
                                        class_names,
                                        f"/workspace/drawed/{border}.jpg")
    
    corrected_samples = []
    label_issue_indices = label_issue_indices.tolist()

    for idx, sample in enumerate(labels):
        sample["image"] = sample["image"].replace(dataset_path, "")

        if idx in label_issue_indices:
            label_idx = label_issue_indices.index(idx)
            corrected_label = predicted_labels[label_idx]

            sample["label"] = class_names[corrected_label]
        
        else:
            sample["label"] = class_names[sample["label"]]

        corrected_samples.append(deepcopy(sample))
    
    with jsonlines.open(os.path.join(dataset_path, "corrected_labels.jsonl"), 'w') as result_file:
        result_file.write_all(corrected_samples)


 

if __name__ == "__main__":
    args = parse_args()
    main(args.config)




        