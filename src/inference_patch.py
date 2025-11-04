import torch
import torch.nn.functional as F

# Generic tiling: useful when dealing with larger-than-train images
@torch.no_grad()
def patch_infer_logits(model, img, patch_size=224, stride=224, device='cuda'):
    """ img: (1,C,H,W) tensor; returns averaged logits over tiled patches """
    _, C, H, W = img.shape
    acc = None; count = 0
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img[:, :, y:y+patch_size, x:x+patch_size].to(device)
            logits = model(patch)
            acc = logits if acc is None else acc + logits
            count += 1
    return acc / max(count, 1)