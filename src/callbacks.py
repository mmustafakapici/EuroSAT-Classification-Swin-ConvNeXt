import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, monitor='val_loss', mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best = None
        self.count = 0
        self.should_stop = False

    def step(self, logs):
        value = logs[self.monitor]
        if self.best is None:
            self.best = value
            return
        improve = (value < self.best - self.min_delta) if self.mode=='min' else (value > self.best + self.min_delta)
        if improve:
            self.best = value
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True

class Checkpoint:
    def __init__(self, path, monitor='val_acc', mode='max'):
        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def step(self, model, logs):
        value = logs[self.monitor]
        if self.best is None:
            self.best = value
            torch.save(model.state_dict(), self.path)
            return
        improve = (value > self.best) if self.mode=='max' else (value < self.best)
        if improve:
            self.best = value
            torch.save(model.state_dict(), self.path)