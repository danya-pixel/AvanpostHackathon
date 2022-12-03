import json
import torch
import torchvision
import numpy as np
from torch import nn
from ml.dataset_working import get_dataloaders, get_dataloader_pred
from ml.train import train_model_default


def get_classes_dict(classes_names):
    return dict(classes_names, range(len(classes_names)))

def finetune_model(classes_names, pth_path, new_data_dir, new_data_name):
    np.random.seed(42)
    torch.seed()
    class_names.append(new_data_name)

    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(1)

    data_dir = 'dataset'
    classes = get_classes_dict(classes_names)
    class_names = {v: k for k, v in classes.items()}

    dataloaders = get_dataloaders(data_dir, classes, new_data_dir, new_data_name)
    model_ft = torchvision.models.resnet18()
    model_ft.load_state_dict(torch.load(pth_path))
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(classes_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model_default(model_ft, device, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

def load_model(classes_names, pth_path, device):
    np.random.seed(42)
    torch.seed()

    model_ft = torchvision.models.resnet18()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(classes_names))
    model_ft.load_state_dict(torch.load(pth_path, map_location=device))
    model_ft.to(device)
    return model_ft

def predict_samples(classes_names, pth_path, new_data_dir):
    np.random.seed(42)
    torch.seed()
    meta = {}
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model_ft = load_model(classes_names, pth_path, device)
    model_ft.eval()
    dataloader = get_dataloader_pred(new_data_dir)
    
    with torch.no_grad():
        for i, (inputs, path) in enumerate(dataloader):
                    inputs = inputs.to(device)

                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    for pred, file in zip(preds, path):
                        meta[file] = int(pred.detach().cpu())
    return meta
    

   
    



