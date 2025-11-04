import torch
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

@torch.no_grad()
def eval_confusion_matrix(model, loader, device):
    y_true, y_pred = [], []
    model.eval()
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        y_pred.extend(pred)
        y_true.extend(y.numpy())
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, output_dict=False)
    return cm, report