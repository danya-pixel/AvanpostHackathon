from typing import Dict
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from PIL import Image
import PIL
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
        self, file_lists, labels_set, transform = None):

        self.files = list(file_lists['x'])
        self.labels = list(file_lists['y'])
        self.labels_set = labels_set
        self.transform = transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img_path = self.files[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        #label = self.labels_set[self.all_files_sep[index][-2]]
        label = self.labels[index]
        return img, label

class ImageNetDatasetPred(Dataset):
    def __init__(
        self, file_lists, transform = None):

        self.files = list(file_lists['x'])
        self.labels = list(file_lists['y'])
        self.transform = transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img_path = self.files[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, img_path
        

def get_all_files(data_dir):
    data_paths = []
    data_labels = []
    for r, _, f in os.walk(data_dir):
        for name in f:
            data_paths.append(os.path.join(r, name))
            data_labels.append(r.split('/')[1])
    return data_paths, data_labels

def get_dataloader_pred(data_dir):
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
    input_transform = transforms.Compose([
        transforms.Resize(256, PIL.Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    new_files, _ = get_all_files(data_dir=data_dir)
    new_lables = 0 * len(new_files)
    init_dataframe = pd.DataFrame({'x': new_files, 'y': new_lables})
    dataset_pred = ImageNetDatasetPred(init_dataframe, transform=input_transform)
    dataloader_pred = torch.utils.data.DataLoader(dataset_pred, batch_size=64, shuffle=True, num_workers=4)
    return dataloader_pred

def get_dataloaders(data_dir, classes, new_data_dir, new_data_name):
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
    input_transform = transforms.Compose([
        transforms.Resize(256, PIL.Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    init_files, init_labels = get_all_files(data_dir=data_dir)
    init_labels = [classes[i] for i in init_labels]
    new_files, _ = get_all_files(data_dir=new_data_dir)
    new_labels = classes[new_data_name] * len(new_files)

    init_dataframe = pd.DataFrame({'x': init_files, 'y': init_labels})
    new_dataframe = pd.DataFrame({'x': new_files, 'y':new_labels})
    dataset_list = pd.concat([init_dataframe, new_dataframe])

    train_data, val_test_data = train_test_split(dataset_list, test_size = 0.2, shuffle = True)
    val_data, test_data = train_test_split(val_test_data, test_size = 0.5, shuffle = True)

    dataset_train = ImageNetDataset(train_data, classes, transform = input_transform)
    dataset_val = ImageNetDataset(val_data, classes, transform = input_transform)
    dataset_test = ImageNetDataset(test_data, classes, transform = input_transform)


    image_datasets = {'train':dataset_train, 'val':dataset_val, 'test':dataset_test} #тут надо вставить датасеты

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val', 'test']}
    
    return dataloaders

def predict_dataloader(new_data_paths):
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
    input_transform = transforms.Compose([
        transforms.Resize(256, PIL.Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    dataset_list = pd.read_csv(new_data_paths)
    dataset_pred = ImageNetDataset(data_dir, train_data, classes, transform = input_transform)