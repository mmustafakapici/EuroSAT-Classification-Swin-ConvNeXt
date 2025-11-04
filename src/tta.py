import torch

def apply_tta(model, x):
    # Simple hflip/vflip TTA
    outs = []
    outs.append(model(x))
    outs.append(model(torch.flip(x, dims=[-1])))  # hflip
    outs.append(model(torch.flip(x, dims=[-2])))  # vflip
    return torch.stack(outs, dim=0).mean(0)