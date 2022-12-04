import torchvision
from pathlib import Path
import numpy as np
import torch

from ml.dataset_working import get_dataloaders, get_dataloader_pred
from ml.test import test_model
from ml.train import train_model_tune


def get_classes_dict(classes_names):
    return dict(zip(classes_names, list(range(len(classes_names)))))


def finetune_model(data_dir, classes_names, pth_path, new_data_dir, new_data_name, ood_data_dir=None,
                   ood_data_name=None):
    meta = dict()
    np.random.seed(42)
    torch.seed()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model_ft = load_model_finetune(classes_names, pth_path, device)

    classes_names.append(new_data_name)
    meta['classes'] = classes_names
    meta['pth_path'] = (Path(pth_path).parent / (Path(pth_path).stem + '_fine.pth')).resolve()

    classes = get_classes_dict(classes_names)

    dataloaders = get_dataloaders(data_dir, classes, new_data_dir, new_data_name)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    for name, param in model_ft.named_parameters():
        if not 'fc' in name:
            param.requires_grad = False
    model_ft = train_model_tune(meta, model_ft, device, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                                num_epochs=5, model_path=meta['pth_path'])

    acc, f1 = test_model(model_ft, dataloaders['test'], device)
    meta['acc'], meta['f1'] = float(acc), float(f1)
    return meta


def load_model(classes_names, pth_path, device):
    np.random.seed(42)
    torch.seed()

    model_ft = torchvision.models.resnet50()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, len(classes_names))
    model_ft.load_state_dict(torch.load(pth_path, map_location=device))
    model_ft.to(device)
    return model_ft


def load_model_finetune(classes_names, pth_path, device):
    np.random.seed(42)
    torch.seed()

    model_ft = torchvision.models.resnet50()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, len(classes_names))
    model_ft.load_state_dict(torch.load(pth_path, map_location=device))
    model_ft.fc = torch.nn.Linear(num_ftrs, len(classes_names) + 1)
    model_ft.to(device)
    return model_ft


def predict_samples(classes_names, pth_path, new_data_dir):
    best_thresholds = [0.1745,
                       0.2794,
                       0.1232,
                       0.2812,
                       0.1567,
                       0.1434,
                       0.2475,
                       0.2341,
                       0.2341,
                       0.3123]
    np.random.seed(42)
    torch.seed()
    meta = {}
    softmax = torch.nn.Softmax(dim=1)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model_ft = load_model(classes_names, pth_path, device)
    model_ft.eval()
    dataloader = get_dataloader_pred(new_data_dir)

    with torch.no_grad():
        for i, (inputs, path) in enumerate(dataloader):
            inputs = inputs.to(device)

            outputs = model_ft(inputs)
            sf_out = softmax(outputs).cpu()
            preds = torch.tensor(
                [
                    -1 if (p < best_thresholds).all() else p.argmax()
                    for p in sf_out.numpy()
                ]
            )
            # _, preds = torch.max(outputs, 1)
            # print(outputs.detach().cpu())
            for pred, file in zip(preds, path):
                meta[file] = int(pred.detach().cpu())
    return meta


def predict_samples_old(classes_names, pth_path, new_data_dir):
    np.random.seed(42)
    torch.seed()
    meta = {}
    softmax = torch.nn.Softmax(dim=1)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model_ft = load_model(classes_names, pth_path, device)
    model_ft.eval()
    dataloader = get_dataloader_pred(new_data_dir)

    with torch.no_grad():
        for i, (inputs, path) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(softmax(outputs), 1)
            for pred, file in zip(preds, path):
                meta[file] = int(pred.detach().cpu())
    return meta
