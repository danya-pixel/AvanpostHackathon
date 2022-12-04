import pandas as pd
from typing import Dict
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
from preprocessing import get_train_transforms, get_test_transforms
import PIL
import numpy as np
import torch
import os
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class ImageNetDataset(Dataset):
    def __init__(
            self, file_lists, labels_set, transform=None):

        self.files = list(file_lists['x'])
        self.labels = list(file_lists['y'])
        self.labels_set = labels_set
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(image=img)

        #label = self.labels_set[self.all_files_sep[index][-2]]
        label = self.labels[index]
        return img['image'], label


class ImageNetDatasetPred(Dataset):
    def __init__(
            self, file_lists, transform=None):

        self.files = list(file_lists['x'])
        self.labels = list(file_lists['y'])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(image=img)
        return img['image'], img_path

def get_all_files(data_dir):
    data_paths = []
    data_labels = []
    for r, _, f in os.walk(data_dir):
        for name in f:
            if '.gitkeep' == name:
                continue
            if name.startswith('.'):
                continue
            name = Path(name)
            r = Path(r)
            filepath = r/name
            label = filepath.parents[0].name
            data_paths.append(str(filepath.resolve()))
            data_labels.append(label)
    return data_paths, data_labels



def get_dataloader_pred(data_dir):
    NUM_WORKERS = 1
    new_files, _ = get_all_files(data_dir=data_dir)
    new_lables = 0 * len(new_files)
    init_dataframe = pd.DataFrame({'x': new_files, 'y': new_lables})
    dataset_pred = ImageNetDatasetPred(
        init_dataframe, transform=get_test_transforms())
    dataloader_pred = torch.utils.data.DataLoader(
        dataset_pred, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
    return dataloader_pred


def get_dataloaders(data_dir, classes, new_data_dir, new_data_name):
    init_files, init_labels = get_all_files(data_dir=data_dir)
    init_labels = [classes[i] for i in init_labels]
    new_files, _ = get_all_files(data_dir=new_data_dir)
    new_labels = classes[new_data_name]

    init_dataframe = pd.DataFrame({'x': init_files, 'y': init_labels})
    new_dataframe = pd.DataFrame({'x': new_files, 'y': new_labels})
    dataset_list = pd.concat([init_dataframe, new_dataframe])

    train_data, val_test_data = train_test_split(
        dataset_list, test_size=0.2, shuffle=True)
    val_data, test_data = train_test_split(
        val_test_data, test_size=0.5, shuffle=True)

    dataset_train = ImageNetDataset(
        train_data, classes, transform=get_train_transforms())
    dataset_val = ImageNetDataset(val_data, classes, transform=get_train_transforms())
    dataset_test = ImageNetDataset(
        test_data, classes, transform=get_test_transforms())

    image_datasets = {'train': dataset_train, 'val': dataset_val,
                      'test': dataset_test}  # тут надо вставить датасеты

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val', 'test']}

    return dataloaders

def get_dataloaders_train(data_dir, classes):
    init_files, init_labels = get_all_files(data_dir=data_dir)
    init_labels = [classes[i] for i in init_labels]

    dataset_list = pd.DataFrame({'x': init_files, 'y': init_labels})

    train_data, val_test_data = train_test_split(
        dataset_list, test_size=0.2, shuffle=True)
    val_data, test_data = train_test_split(
        val_test_data, test_size=0.5, shuffle=True)

    dataset_train = ImageNetDataset(
        train_data, classes, transform=get_train_transforms())
    dataset_val = ImageNetDataset(val_data, classes, transform=get_train_transforms())
    dataset_test = ImageNetDataset(
        test_data, classes, transform=get_test_transforms())

    image_datasets = {'train': dataset_train, 'val': dataset_val,
                      'test': dataset_test}  # тут надо вставить датасеты

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                  shuffle=True, num_workers=32)
                   for x in ['train', 'val', 'test']}

    return dataloaders
