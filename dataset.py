import os
import torch
from torch.utils.data import Dataset as dts
import json
import jsonlines
from PIL import Image

class Dataset(dts):
    def __init__(self, 
                  annotations_path: str, 
                  names: dict, 
                  transform=None,
                  return_path=False) -> None:

        self.return_path = return_path

        self.names = names
        self.transform = transform        

        self.paths = []
        self.labels = []

        with jsonlines.open(annotations_path) as reader:
            for sample in reader:
                self.paths.append(os.path.join(os.path.dirname(annotations_path).replace("folds", ''), sample["image"]))
                self.labels.append(sample["label"])

        self.label2idx = {label:idx for idx, label in enumerate(names)} 

        
    def __getitem__(self, idx: int):
        image_path = self.paths[idx]

        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image)
        label = torch.tensor(self.label2idx[self.labels[idx]])
        
        if self.return_path:
            return tensor, label, image_path
        return tensor, label



    def __len__(self,):
        return len(self.paths)





    

        
