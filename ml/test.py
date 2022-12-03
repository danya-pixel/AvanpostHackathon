from torch import nn
import torch
from sklearn.metrics import f1_score

def test_model(model, dataloader, device):
    model.eval()  # Evaluation mode
   
    correct_samples = 0
    total_samples = 0
    y_true = []
    y_pred = []
    y_probs = [] 

    for i, (img, labels) in enumerate(dataloader):
        with torch.no_grad():
            prediction = model(img.to(device)).cpu()
            _, indices = torch.max(prediction, axis=1)
            y_pred.extend(indices)
            y_true.extend(labels)
            y_probs.extend(prediction.numpy())
            correct_samples += torch.sum(indices == labels)
            total_samples += labels.shape[0]
    
    f1 = f1_score(y_true, y_pred, average = 'weighted')
    accuracy = correct_samples/total_samples
    return accuracy, f1