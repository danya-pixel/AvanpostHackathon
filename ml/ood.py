import torch
import numpy as np
import tqdm


from torch import nn
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize


def test_model_ood(model, dataloader, threshold, device):
    model.eval()  # Evaluation mode

    softmax = nn.Softmax(dim=-1)

    correct_samples = 0
    total_samples = 0
    y_true = []
    y_pred = []
    y_probs = []

    for i, (img, labels) in enumerate(dataloader):
        with torch.no_grad():
            prediction = model(img.to(device)).cpu()
            pred_probs = softmax(prediction).cpu()
            indices = torch.tensor(
                [
                    -1 if (p < threshold).all() else p.argmax()
                    for p in pred_probs.numpy()
                ]
            )
            y_pred.extend(indices)
            y_true.extend(labels)
            y_probs.extend(pred_probs.numpy())
            correct_samples += torch.sum(indices == labels)
            total_samples += labels.shape[0]

    f1 = f1_score(y_true, y_pred, average="weighted")
    accuracy = correct_samples / total_samples
    output = {
        "accuracy": accuracy,
        "f1": f1,
        "y_pred": np.array(y_pred),
        "y_true": np.array(y_true),
        "y_probs": np.array(y_probs),
    }

    return output


def find_smart_thresholds(model, device, dataloaders, compute_accuracy, n_classes=10):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    best_thresholds = []
    data = compute_accuracy(model, dataloaders["val"], 0, device)
    data_test = compute_accuracy(model, dataloaders["test"], 0, device)

    y_test = label_binarize(data["y_true"], classes=list(range(n_classes)))

    for i in tqdm.tqdm(range(n_classes)):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], data["y_probs"][:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    del y_test
    for i in range(n_classes):
        t = np.argmax(tpr[i] - fpr[i])
        best_thresholds.append(thresholds[i][t])

    data_val = compute_accuracy(model, dataloaders["val"], best_thresholds, device)
    data_test = compute_accuracy(model, dataloaders["test"], best_thresholds, device)

    results = {
        "init_val_f1": data["f1"],
        "init_val_acc": data["accuracy"],
        "init_test_f1": data_test["f1"],
        "init_test_acc": data_test['accuracy'],
        "best_threshold": best_thresholds,
        "val_accuracy": data_val["accuracy"],
        "val_f1": data_val["f1"],
        "test_accuracy": data_test["accuracy"],
        "test_f1": data_test["f1"],
    }

    return results
