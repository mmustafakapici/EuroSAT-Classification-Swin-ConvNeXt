import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'): 
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1-pt)**self.gamma) * ce
        return loss.mean() if self.reduction=='mean' else loss.sum()

# NOTE: Dice/Lovasz are segmentation-centric; provided as experimental variants.
class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__(); self.eps=eps
    def forward(self, logits, target):
        # convert to one-hot
        num_classes = logits.size(1)
        target_oh = F.one_hot(target, num_classes).float()
        probs = F.softmax(logits, dim=1)
        inter = (probs*target_oh).sum(dim=0)
        union = probs.sum(dim=0) + target_oh.sum(dim=0)
        dice = (2*inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()

# Placeholder Lovasz (multi-class). For thoroughness you may swap with a dedicated lib.
class LovaszSoftmaxLike(nn.Module):
    def forward(self, logits, target):
        return F.cross_entropy(logits, target)  # fallback

def build_loss(name: str, class_weights=None, focal_gamma=2.0):
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == 'focal':
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    if name == 'soft_dice':
        return SoftDiceLoss()
    if name == 'lovasz_softmax':
        return LovaszSoftmaxLike()
    raise ValueError(f"Unknown loss: {name}")