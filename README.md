
# EuroSAT – Swin/ConvNeXt (PyTorch, Transfer Learning)

EuroSAT RGB veri seti üzerinde **Swin Transformer** ve **ConvNeXt** ile transfer learning tabanlı görüntü sınıflandırma.
Proje; **mixed precision**, **early stopping & checkpointing**, **LR scheduler (Cosine / ReduceLROnPlateau)**, **farklı loss fonksiyonları (CrossEntropy, Focal)**, **TTA**, **patch-based inference**, **confusion matrix** ve **Gradio inference UI** içerir. ROCm/AMD GPU ile de uyumludur.

## ✨ Özellikler

* **Modeller**: Swin Transformer / ConvNeXt (timm)
* **Transfer Learning**: ImageNet ön-eğitimli
* **Augmentasyon**: Albumentations (train/val ayrık), TTA (isteğe bağlı)
* **Class Imbalance**: Weighted sampler veya Focal Loss
* **Eğitim İyileştirmeleri**: Mixed precision (AMP), early stopping, en iyi checkpoint kaydı
* **LR Stratejileri**: Cosine annealing, ReduceLROnPlateau, warmup ile kombinasyon
* **Değerlendirme**: Accuracy, confusion matrix; yanlış sınıf örnekleri analizi
* **Patch-based Inference**: Büyük görselleri parçalayarak tahmin
* **Gradio UI**: Tek görselden tahmin (TTA/patch seçenekleri)
* **DDP (Multi-GPU)**: `torchrun` ile dağıtık eğitim

---

## 📦 Proje Yapısı (özet)

```
.
├── conf/
│   ├── config.yaml           # ana konfig
│   └── labels.txt            # sınıf adları (EuroSAT için 10 sınıf)
├── data/                     # (make download ile dolar) data/EuroSAT_RGB/...
├── outputs/
│   ├── checkpoints/          # en iyi modeller
│   └── tensorboard/          # TB logları
├── gradio_app/
│   └── app.py                # Gradio arayüzü
├── scripts/
│   ├── download_eurosat.py   # KaggleHub ile RGB klasörünü kopyalar
│   ├── cuda_info.py          # ROCm/CUDA bilgi çıktısı
│   └── make_samples.py       # samples/ klasörü üretir
├── src/
│   ├── dataset.py            # Custom Dataset/aug boru hattı
│   ├── models.py             # Swin/ConvNeXt builder (timm)
│   ├── engine.py             # train/validate döngüsü (AMP, TTA entegrasyonu)
│   ├── losses.py             # CrossEntropy, Focal vs.
│   ├── tta.py                # basit TTA çağrıları
│   └── inference_patch.py    # patch-based inference
├── train.py                  # eğitim komutu
├── inference.py              # tek görsel inference CLI
├── Makefile                  # kısayol komutlar
├── requirements-rocm.txt     # ROCm uyumlu PyTorch (rocm6.4 index)
└── requirements.txt          # diğer bağımlılıklar (PyPI)
```

---

## 🚀 Kurulum

### 0) Ortam

Conda önerilir (Python 3.10):

```bash
conda create -n eurosat python=3.10 -y
conda activate eurosat
python -m pip install --upgrade pip
```

### 1) Bağımlılıklar

**ROCm tabanlı PyTorch** (AMD/ROCm için) ve diğerleri ayrı dosyalarda tutulur:

```bash
# PyTorch ROCm (rocm6.4 index’i)
pip install -r requirements-rocm.txt

# Diğer bağımlılıklar (PyPI) – gradio 5.x dahil
pip install -r requirements.txt
```

> NVIDIA/CUDA kullanıyorsan kendi CUDA tekerlerinle `torch/vision/audio` kurup `requirements-rocm.txt` adımını atla.

---

## 📥 Veri Seti (EuroSAT RGB)

```bash
make download
# KaggleHub ile indirir, sadece EuroSAT (RGB) klasörünü data/EuroSAT_RGB/ içine kopyalar.
```

Örnek görseller:

```bash
make samples K=3     # her sınıftan 3 görseli samples/ içine kopyalar
```

---

## 🏃 Eğitim

```bash
# Tek GPU
make train
# → python train.py --cfg conf/config.yaml
```

**DDP (çoklu GPU)**:

```bash
make ddp GPUS=2
# → torchrun --nproc_per_node=2 train.py --cfg conf/config.yaml
```

> `conf/config.yaml` içinde: batch size, img_size, scheduler/loss/TTA/AMP/early-stopping gibi ayarları yönetebilirsin.
> EuroSAT RGB görüntüleri 64×64 olduğundan **val pipeline’da Resize(224,224)** kullanılır.

---

## 📊 TensorBoard

```bash
make tb
# → tensorboard --logdir outputs/tensorboard
# http://localhost:6006
```

Varsayılan loglar: `train/acc`, `train/loss`, `val/acc`, `val/loss`, `opt/lr`.
İstersen confusion matrix, misclassified örnek görüntüleri ve hparams karşılaştırması eklenebilir.

---

## 🔎 Inference

### CLI (tek görsel)

```bash
make infer IMG=path/to/image.jpg
# → inference.py --cfg conf/config.yaml --ckpt outputs/checkpoints/best.pt --img ...
```

### Gradio UI

```bash
make gradio
# http://0.0.0.0:7860
# TTA/patch-based seçenekleri, top-k tablo, latency gösterimi
```

> Eğer `ModuleNotFoundError: src` görürsen Makefile’daki `gradio` hedefi `PYTHONPATH=.` ile çağrılıyor olmalı.

---

## ⚙️ Önemli Ayarlar (config.yaml)

* **Model**: `model.arch` (`swin_tiny_patch4_window7_224`, `convnext_tiny.fb_in22k` vb.)
* **Loss**: `loss.name` (`cross_entropy` / `focal`), `loss.focal_gamma`
* **Optim**: `optim.name` (AdamW), `optim.lr`, `optim.weight_decay`
* **Scheduler**: `scheduler.name` (`cosine`, `plateau`), warmup adımı/parametreler
* **AMP**: `train.mixed_precision: true` (PyTorch 2.0+ için `torch.amp`)
* **Early Stopping**: `train.early_stopping.patience`, `min_delta`
* **TTA**: `tta.enabled: true` (h/v flip gibi basit TTA)
* **Patch Inference**: `inference.patch.enabled`, `patch_size`, `stride`

---

## 🧪 Patch-based Inference

Büyük görüntüler için model input boyutuna (ör. 224) **parçalayarak** tahmin.
`src/inference_patch.py` ile patch logits’leri **combine** edilip sınıf skorları üretilir. Gradio arayüzünden açıp kapatılabilir.

---

## 🧰 Makefile Hedefleri

| Hedef                | Açıklama                                                      |
| -------------------- | ------------------------------------------------------------- |
| `make install`       | ROCm PyTorch + diğer bağımlılıkları kur (iki dosyayı da okur) |
| `make download`      | EuroSAT RGB indir/kopyala                                     |
| `make samples K=3`   | Her sınıftan K görseli `samples/` içine kopyala               |
| `make train`         | Eğitim başlat                                                 |
| `make ddp GPUS=N`    | Çoklu GPU DDP eğitim (`torchrun`)                             |
| `make infer IMG=...` | Tek görsel inference                                          |
| `make gradio`        | Gradio UI                                                     |
| `make tb`            | TensorBoard başlat                                            |
| `make cuda-info`     | ROCm/CUDA/PyTorch sürüm bilgisi                               |
| `make clean`         | Çıktıları ve cache’leri temizle                               |

---

## 🛠️ Sorun Giderme

* **Albumentations CropSizeError (64 vs 224)**:
  Val pipeline’da `Resize(224,224)` kullan (proje default’u bu; özelleştirdiysen geri al).
* **`ModuleNotFoundError: src` (Gradio)**:
  Makefile `gradio` hedefi `PYTHONPATH=.` ile çağrılıyor olmalı.
* **ROCm/NVIDIA karışıklığı**:
  ROCm için `requirements-rocm.txt`; CUDA için `torch/vision/audio`’yu kendi CUDA index’inle kur.
* **Gradio 5.x bağımlılık uyarıları**:
  Bu proje `transformers/tokenizers/wandb` kullanmaz; ortamda varsa kaldırabilir veya pin’leyebilirsin.

---

## 📚 Kaynakça

* **EuroSAT:** Helber, P. et al. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* (RGB, 10 sınıf)
* **timm:** [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
* **Albumentations:** [https://github.com/albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)
* **PyTorch AMP (2.x):** `torch.amp.autocast`, `torch.amp.GradScaler`

---

## 📄 Lisans

MIT — dilediğin gibi kullan, katkı PR’ları memnuniyetle.

---
## License
MIT © 2025 Mustafa Kapıcı / AILAYZER LTD
