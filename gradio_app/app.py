import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # proje kökünü sys.path'e ekle

import gradio as gr
import torch, yaml
from pathlib import Path
from PIL import Image
import numpy as np
import time

from src.models import build_model
from src.tta import apply_tta
from src.dataset import build_aug
from src.inference_patch import patch_infer_logits

def load_labels(labels_path: str):
    p = Path(labels_path)
    if not p.exists():
        return [
            "AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial",
            "Pasture","PermanentCrop","Residential","River","SeaLake"
        ]
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

class Predictor:
    def __init__(self, cfg_path, ckpt_path, labels_path, use_tta=True, device=None):
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.labels = load_labels(labels_path)
        self.use_tta = use_tta
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._load()

    def _load(self):
        self.cfg = yaml.safe_load(open(self.cfg_path))
        self.model = build_model(
            self.cfg['model']['arch'],
            self.cfg['model']['num_classes'],
            self.cfg['model']['pretrained'],
            self.cfg['model']['compile']
        ).to(self.dev)
        # Güvenli load: weights_only=True (mümkünse); değilse fallback
        try:
            state = torch.load(self.ckpt_path, map_location=self.dev, weights_only=True)  # PyTorch >=2.4+
        except TypeError:
            state = torch.load(self.ckpt_path, map_location=self.dev)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.aug = build_aug(self.cfg['aug']['val'], self.cfg['data']['img_size'])

    @torch.no_grad()
    def predict(self, img: Image.Image, use_tta: bool=None, patch_mode: bool=False, patch_size: int=224, stride: int=224):
        if img is None:
            return "No image", []  # 2 çıktı: (metin, topk tablo satırları)
        use_tta = self.use_tta if use_tta is None else use_tta
        npt = np.array(img.convert('RGB'))
        x = self.aug(image=npt)['image'].unsqueeze(0).to(self.dev)

        start = time.time()
        # ROCm için yeni API: autocast('cuda')
        ctx = torch.amp.autocast('cuda') if self.dev == 'cuda' else torch.no_grad()
        with ctx:
            if patch_mode:
                logits = patch_infer_logits(self.model, x, patch_size=int(patch_size), stride=int(stride), device=self.dev)
            else:
                logits = apply_tta(self.model, x) if (use_tta and self.cfg.get('tta', {}).get('enabled', False)) else self.model(x)

        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        conf, idx = torch.max(probs, dim=0)
        topk = min(5, probs.numel())
        vals, inds = torch.topk(probs, k=topk)
        elapsed = (time.time() - start) * 1000.0

        pred_label = self.labels[idx.item()] if idx.item() < len(self.labels) else str(idx.item())
        top_rows = [(self.labels[i] if i < len(self.labels) else str(i), float(v)) for v,i in zip(vals.tolist(), inds.tolist())]
        return f"{pred_label} ({conf:.3f}) · {elapsed:.1f} ms", top_rows

def build_ui(default_cfg="conf/config.yaml", default_ckpt="outputs/checkpoints/best.pt", labels_path="conf/labels.txt"):
    predictor = Predictor(default_cfg, default_ckpt, labels_path, use_tta=True)
    with gr.Blocks(title="EuroSAT Inference") as demo:
        gr.Markdown("# EuroSAT – Swin/ConvNeXt Inference")
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Input Image")
                use_tta = gr.Checkbox(value=True, label="Use TTA (h/v flip)")
                patch_mode = gr.Checkbox(value=False, label="Patch-based inference")
                patch_size = gr.Slider(64, 512, value=224, step=32, label="Patch size")
                stride = gr.Slider(32, 512, value=224, step=32, label="Stride")
                cfg_path = gr.Textbox(value=default_cfg, label="Config Path", scale=3)
                ckpt_path = gr.Textbox(value=default_ckpt, label="Checkpoint Path", scale=3)
                reload_btn = gr.Button("Reload Model")
                run = gr.Button("Predict", variant="primary")
            with gr.Column():
                pred = gr.Textbox(label="Prediction · latency", interactive=False)
                topk = gr.Dataframe(headers=["Class","Confidence"], datatype=["str","number"], label="Top-k", interactive=False)

        def _predict(img, use_tta, patch_mode, patch_size, stride):
            return predictor.predict(img, use_tta, patch_mode, int(patch_size), int(stride))

        def _reload(cfg_path_v, ckpt_path_v):
            predictor.cfg_path = cfg_path_v
            predictor.ckpt_path = ckpt_path_v
            predictor._load()
            return gr.update(value="Model reloaded ✓")

        run.click(_predict, inputs=[img, use_tta, patch_mode, patch_size, stride], outputs=[pred, topk])  # <<< 2 çıktı
        reload_btn.click(_reload, inputs=[cfg_path, ckpt_path], outputs=[pred])
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860 , share=True)
