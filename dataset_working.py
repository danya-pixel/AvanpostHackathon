from typing import Dict
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from PIL import Image
import torch
import os
import glob
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

#from preproccesing import get_test_transforms, get_train_transforms
import pandas as pd


class ImageNetDataset(Dataset):
    def __init__(
        self, data_dir, file_lists, labels_set, transform = None):

        
        # all_files = glob.glob(data_dir + '/**/*', recursive=True)
        # self.all_files_paths = [file for file in all_files if len(os.path.normpath(file).split(os.sep)) == 3]
        # self.all_files_sep = [os.path.normpath(file).split(os.sep) for file in self.all_files_paths]
        self.data_dir = data_dir
        self.all_files_names = list(file_lists['X_name'])
        self.all_files_dirs = list(file_lists['Y_name'])
        self.all_files_labels = list(file_lists['Y'])


        self.labels_set = labels_set
        self.transform = transform

    def __len__(self):
        return len(self.all_files_names)
    
    def __getitem__(self, index):
        #img_path = self.all_files_paths[index]
        img_path = os.path.join(self.data_dir, self.all_files_dirs[index], self.all_files_names[index])
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        #label = self.labels_set[self.all_files_sep[index][-2]]
        label = self.all_files_labels[index]
        return img, label
        





