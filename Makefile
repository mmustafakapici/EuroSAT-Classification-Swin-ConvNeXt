# ==== Config ====
CONFIG   ?= conf/config.yaml
CKPT     ?= outputs/checkpoints/best.pt
IMG      ?= path/to/sample.jpg
GPUS     ?= 2
PY       ?= python
PIP      ?= pip
TB_DIR   ?= outputs/tensorboard
DATA_DIR ?= data/EuroSAT_RGB
OUT_DIR  ?= samples
K        ?= 1

SHELL := /bin/bash
.ONESHELL:

# ==== Help ====
.PHONY: help
help:
	@echo "Targets:"
	@echo "  make install              # requirements.txt kur"
	@echo "  make download             # EuroSAT RGB indir + data klasörüne kopyala"
	@echo "  make train                # modeli eğit (CONFIG=$(CONFIG))"
	@echo "  make tb                   # TensorBoard aç (dir=$(TB_DIR))"
	@echo "  make infer IMG=...        # tek görselde inference (CKPT=$(CKPT))"
	@echo "  make ddp GPUS=2           # çoklu GPU DDP eğitim (torchrun)"
	@echo "  make freeze               # requirements-lock.txt üret"
	@echo "  make env                  # ortam ve sürüm bilgisi"
	@echo "  make clean                # çıktı/önbellek temizle"
	@echo "  make cuda-info            # CUDA/ROCm bilgisi"
	@echo "  make samples K=3          # her sınıftan K görseli samples/ klasörüne kopyala"
	@echo "  make gradio               # Gradio inference UI"

# ==== Setup ====
.PHONY: install
install:
	$(PIP) install -r requirements-rocm.txt
	$(PIP) install -r requirements.txt

.PHONY: env
env:
	$(PY) -V
	$(PIP) -V
	$(PY) -c "import torch, timm; print('torch:', torch.__version__, ' cuda:', torch.cuda.is_available()); print('timm:', getattr(timm, '__version__', 'unknown'))"

# ==== CUDA/ROCm Info ====
.PHONY: cuda-info
cuda-info:
	$(PY) scripts/cuda_info.py

# ==== Data ====
.PHONY: download
download:
	$(PY) scripts/download_eurosat.py --dst $(DATA_DIR)

# ==== Samples ====
.PHONY: samples
samples:
	$(PY) scripts/make_samples.py --data-dir $(DATA_DIR) --out-dir $(OUT_DIR) --k $(K)

# ==== Train / TB / Infer ====
.PHONY: train
train:
	$(PY) train.py --cfg $(CONFIG)

.PHONY: tb
tb:
	tensorboard --logdir $(TB_DIR)

.PHONY: infer
infer:
	@if [ -z "$(IMG)" ]; then echo "IMG yolu gerekli: make infer IMG=path/to.jpg"; exit 2; fi
	$(PY) inference.py --cfg $(CONFIG) --ckpt $(CKPT) --img "$(IMG)"

# ==== DDP ====
.PHONY: ddp
ddp:
	torchrun --nproc_per_node=$(GPUS) train.py --cfg $(CONFIG)

# ==== Utilities ====
.PHONY: freeze
freeze:
	$(PIP) freeze > requirements-lock.txt
	@echo "✔ requirements-lock.txt oluşturuldu."

.PHONY: clean
clean:
	rm -rf outputs/* *.pt **/__pycache__ .pytest_cache
	@echo "✔ Temizlendi."

# ==== Gradio ====
..PHONY: gradio
gradio:
	PYTHONPATH=. $(PY) gradio_app/app.py
