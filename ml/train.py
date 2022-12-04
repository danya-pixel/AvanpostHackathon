import time
import copy
import torch
import numpy as np
import torchvision
from pathlib import Path

from tqdm import tqdm

from ml.test import test_model
from ml.dataset_working import get_dataloaders_train
from torchvision.models import resnet50, ResNet50_Weights


def train_model(
        meta,
        model,
        device,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        num_epochs=10,
        model_path="model.pth",
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    softmax = torch.nn.Softmax(dim=1)
    meta["epoch_loss"] = []
    meta["epoch_acc"] = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(softmax(outputs), 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            meta["epoch_loss"].append(epoch_loss)
            meta["epoch_acc"].append(float(epoch_acc.detach().cpu()))

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_path)

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_path)
    return model


def train_model_tune(
        meta,
        model,
        device,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        num_epochs=10,
        model_path="model.pth",
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    softmax = torch.nn.Softmax(dim=1)
    meta["epoch_loss"] = []
    meta["epoch_acc"] = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(softmax(outputs), 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            meta["epoch_loss"].append(epoch_loss)
            meta["epoch_acc"].append(float(epoch_acc.detach().cpu()))

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_path)

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_path)
    return model


if __name__ == "__main__":
    pth_path = "/home/danya-sakharov/AvanpostHackathon/ml/model.pth"
    data_dir = "/home/danya-sakharov/AvanpostHackathon/ml/dataset"
    classes = {
        "trucks": 0,
        "minibus": 1,
        "ski": 2,
        "dump_trucks": 3,
        "bicycles": 4,
        "snowboard": 5,
        "tractor": 6,
        "trains": 7,
        "gazon": 8,
        "horses": 9,
    }
    meta = dict()

    np.random.seed(42)
    torch.seed()

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    model_ft = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, len(classes))
    model_ft.to(device)

    meta["classes"] = classes
    meta["pth_path"] = (
            Path(pth_path).parent / (Path(pth_path).stem + "50.pth")
    ).resolve()

    dataloaders = get_dataloaders_train(data_dir, classes)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1
    )

    model_ft = train_model(
        meta,
        model_ft,
        device,
        dataloaders,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=20,
        model_path=meta["pth_path"],
    )
    acc, f1 = test_model(model_ft, dataloaders["test"], device)
    meta["acc"], meta["f1"] = float(acc), float(f1)
