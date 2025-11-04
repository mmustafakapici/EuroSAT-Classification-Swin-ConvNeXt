import torch
from tqdm import tqdm
from torch.amp import autocast


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None, grad_clip_norm=None):
    model.train(); total=0; correct=0; loss_sum=0.0
    pbar = tqdm(loader, desc='train')
    for x,y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast('cuda'):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        preds = logits.argmax(1)
        correct += (preds==y).sum().item(); total += y.size(0)
        loss_sum += loss.item()*y.size(0)
        pbar.set_postfix(loss=loss_sum/total, acc=correct/total)
    return loss_sum/total, correct/total

@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval(); total=0; correct=0; loss_sum=0.0
    for x,y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        preds = logits.argmax(1)
        correct += (preds==y).sum().item(); total += y.size(0)
        loss_sum += loss.item()*y.size(0)
    return loss_sum/total, correct/total