from typing import Dict
from torch.utils.data import Dataset
import cv2
from PIL import Image
import torch
import os
import glob
import pandas as pd

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

#from preproccesing import get_test_transforms, get_train_transforms



class ImageNetDataset(Dataset):
    def __init__(
        self, data_dir, labels_set, transform = None):

        
        all_files = glob.glob(data_dir + '/**/*', recursive=True)
        self.all_files_paths = [file for file in all_files if len(os.path.normpath(file).split(os.sep)) == 3]
        self.all_files_sep = [os.path.normpath(file).split(os.sep) for file in self.all_files_paths]
        self.labels_set = labels_set
        self.transform = transform

    def __len__(self):
        return len(self.all_files_paths)
    
    def __getitem__(self, index):
        img_path = self.all_files_paths[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels_set[self.all_files_sep[index][-2]]
        return img, label