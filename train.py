import os
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from src.utils import set_seed, ensure_dir, device
from src.dataset import load_paths_labels, stratified_split, EuroSATDataset, build_aug
from src.models import build_model
from src.losses import build_loss
from src.engine import train_one_epoch, validate
from src.callbacks import EarlyStopping, Checkpoint
from src.evaluate import run_eval, save_misclassified
from torch.amp import GradScaler

def build_scheduler(name, optimizer, cfg):
    if name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=cfg['cosine']['T_max'], eta_min=cfg['cosine']['eta_min'])
    if name == 'reduce_on_plateau':
        p = cfg['reduce_on_plateau']
        return ReduceLROnPlateau(optimizer, factor=p['factor'], patience=p['patience'], min_lr=p['min_lr'])
    return None

def main(cfg_path='conf/config.yaml'):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['train']['seed'])
    out_dir = Path(cfg['paths']['output_dir']); ensure_dir(out_dir)
    ensure_dir(cfg['paths']['ckpt_dir']); ensure_dir(cfg['paths']['tb_log_dir']); ensure_dir(cfg['paths']['miscls_dir'])

    dev = device()

    # ==== Data paths & split ====
    root = Path(cfg['paths']['data_dir'])
    paths, labels, classes = load_paths_labels(root)
    tr_p, tr_y, va_p, va_y, te_p, te_y = stratified_split(paths, labels, cfg['data']['val_size'], cfg['data']['test_size'], cfg['train']['seed'])

    # ==== Aug ====
    img_size = cfg['data']['img_size']
    aug_tr = build_aug(cfg['aug'].get('train'), img_size)
    aug_va = build_aug(cfg['aug'].get('val'), img_size)

    ds_tr = EuroSATDataset(tr_p, tr_y, aug_tr)
    ds_va = EuroSATDataset(va_p, va_y, aug_va)

    # ==== Imbalance handling ====
    sampler = None
    if cfg['data']['class_imbalance'] == 'weighted_sampler':
        import numpy as np
        class_counts = np.bincount(tr_y)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[y] for y in tr_y]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=cfg['data']['batch_size'], shuffle=(sampler is None), sampler=sampler,
                       num_workers=cfg['data']['num_workers'], pin_memory=cfg['data']['pin_memory'])
    dl_va = DataLoader(ds_va, batch_size=cfg['data']['batch_size'], shuffle=False,
                       num_workers=cfg['data']['num_workers'], pin_memory=cfg['data']['pin_memory'])

    # ==== Model ====
    model = build_model(cfg['model']['arch'], cfg['model']['num_classes'], cfg['model']['pretrained'], cfg['model']['compile'])
    model = model.to(dev)

    # ==== Loss ====
    class_weights = None
    if cfg['data']['class_imbalance'] == 'class_weights' and cfg['loss']['class_weights'] == 'auto':
        import numpy as np
        class_counts = np.bincount(tr_y)
        w = (class_counts.sum() / (len(class_counts) * (class_counts + 1e-6)))
        class_weights = torch.tensor(w, dtype=torch.float32).to(dev)
    loss_fn = build_loss(cfg['loss']['name'], class_weights, cfg['loss']['focal_gamma'])

    # ==== Optim & Sched ====
    opt = AdamW(model.parameters(), lr=cfg['optim']['lr'], weight_decay=cfg['optim']['weight_decay'], betas=tuple(cfg['optim']['betas']))
    sched = build_scheduler(cfg['scheduler']['name'], opt, cfg['scheduler'])

    # ==== AMP ====
    scaler = GradScaler('cuda', enabled=cfg['train']['mixed_precision'])

    # ==== Callbacks ====
    es = EarlyStopping(**cfg['train']['early_stopping'], mode='min' if cfg['train']['early_stopping']['monitor']=='val_loss' else 'max')
    ckpt_path = os.path.join(cfg['paths']['ckpt_dir'], 'best.pt')
    ckpt = Checkpoint(ckpt_path, **cfg['train']['checkpoint'])

    # ==== TB ====
    writer = SummaryWriter(cfg['paths']['tb_log_dir'])

    # ==== Training loop ====
    best_val = -1
    for epoch in range(cfg['train']['epochs']):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, opt, loss_fn, dev, scaler if cfg['train']['mixed_precision'] else None, cfg['train']['grad_clip_norm'])
        va_loss, va_acc = validate(model, dl_va, loss_fn, dev)

        # sched
        if sched is not None:
            if isinstance(sched, ReduceLROnPlateau):
                sched.step(va_loss)
            else:
                sched.step()

        # logs
        writer.add_scalar('Loss/train', tr_loss, epoch)
        writer.add_scalar('Loss/val', va_loss, epoch)
        writer.add_scalar('Acc/train', tr_acc, epoch)
        writer.add_scalar('Acc/val', va_acc, epoch)

        logs = {'val_loss': va_loss, 'val_acc': va_acc}
        es.step(logs)
        ckpt.step(model, logs)

        print(f"Epoch {epoch+1}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if es.should_stop:
            print('Early stopping triggered.')
            break

    # ==== Final eval & artifacts ====
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    cm, report = run_eval(model, dl_va, dev)
    print('Confusion Matrix:\n', cm)
    print('Classification Report:\n', report)

    save_misclassified(model, dl_va, classes, cfg['paths']['miscls_dir'], dev)
    writer.close()

if __name__ == '__main__':
    main()