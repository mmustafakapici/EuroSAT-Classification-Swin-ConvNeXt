import yaml
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from src.models import build_model
from src.tta import apply_tta

@torch.no_grad()
def predict_image(cfg_path, ckpt_path, img_path, use_tta=True):
    cfg = yaml.safe_load(open(cfg_path))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg['model']['arch'], cfg['model']['num_classes'], cfg['model']['pretrained'], cfg['model']['compile']).to(dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()

    # simple eval pipeline matching val aug (Resize->CenterCrop->Norm)
    from src.dataset import build_aug
    aug = build_aug(cfg['aug']['val'], cfg['data']['img_size'])

    img = np.array(Image.open(img_path).convert('RGB'))
    x = aug(image=img)['image'].unsqueeze(0).to(dev)
    if use_tta and cfg['tta']['enabled']:
        logits = apply_tta(model, x)
    else:
        logits = model(x)
    pred = logits.softmax(1).argmax(1).item()
    return pred

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='conf/config.yaml')
    ap.add_argument('--ckpt', default='outputs/checkpoints/best.pt')
    ap.add_argument('--img', required=True)
    ap.add_argument('--no-tta', action='store_true')
    args = ap.parse_args()
    p = predict_image(args.cfg, args.ckpt, args.img, use_tta=not args.no_tta)
    print('Prediction:', p)