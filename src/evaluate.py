import os
import torch
from torch.utils.data import DataLoader
from .metrics import eval_confusion_matrix
from .utils import ensure_dir
import numpy as np
from PIL import Image

@torch.no_grad()
def save_misclassified(model, loader, classes, out_dir, device):
    ensure_dir(out_dir)
    model.eval()
    idx = 0
    for x,y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(1).cpu()
        y = y.cpu()
        mism = (preds!=y).nonzero().flatten()
        for j in mism:
            img = x[j].cpu().permute(1,2,0).numpy()
            # assuming normalized, denorm might be needed; here we scale back
            img = (img*255).clip(0,255).astype(np.uint8)
            pil = Image.fromarray(img)
            pil.save(os.path.join(out_dir, f"{idx}_pred-{classes[preds[j]]}_gt-{classes[y[j]]}.png"))
            idx += 1

def run_eval(model, val_loader, device):
    cm, report = eval_confusion_matrix(model, val_loader, device)
    return cm, report