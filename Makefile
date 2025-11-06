# ==== Config ====
CONFIG   ?= conf/config.yaml
CKPT     ?= outputs/checkpoints/best.pt
IMG      ?= path/to/sample.jpg
GPUS     ?= 2
TB_DIR   ?= outputs/tensorboard
DATA_DIR ?= data/EuroSAT_RGB
OUT_DIR  ?= samples
K        ?= 1

# ==== Venv Ayarları ====
VENV     ?= .venv
PY       ?= $(VENV)/bin/python
PIP      ?= $(VENV)/bin/pip

SHELL := /bin/bash
.ONESHELL:

# ==== Help ====
.PHONY: help
help:
	@echo "Targets:"
	@echo "  make venv                 # .venv ortamını oluştur"
	@echo "  make install              # venv + requirements-rocm.txt + requirements.txt kur"
	@echo "  make download             # EuroSAT RGB indir + data klasörüne kopyala"
	@echo "  make train                # modeli eğit (CONFIG=$(CONFIG))"
	@echo "  make tb                   # TensorBoard aç (dir=$(TB_DIR))"
	@echo "  make infer IMG=...        # tek görselde inference (CKPT=$(CKPT))"
	@echo "  make ddp GPUS=2           # çoklu GPU DDP eğitim (torchrun)"
	@echo "  make freeze               # requirements-lock.txt üret (venv içinden)"
	@echo "  make env                  # ortam ve sürüm bilgisi (venv)"
	@echo "  make clean                # çıktı/önbellek temizle"
	@echo "  make cuda-info            # CUDA/ROCm bilgisi"
	@echo "  make samples K=3          # her sınıftan K görseli samples/ klasörüne kopyala"
	@echo "  make gradio               # Gradio inference UI"

# ==== Venv Oluşturma ====
..PHONY: venv
venv:
	@if command -v python3 >/dev/null 2>&1; then \
		PYBIN=python3; \
	elif command -v python >/dev/null 2>&1; then \
		PYBIN=python; \
	else \
		echo "❌ Python bulunamadı. Lütfen python3 kur."; \
		exit 1; \
	fi; \
	$$PYBIN -m venv $(VENV); \
	$(VENV)/bin/python -m pip install --upgrade pip; \
	echo "✔ Venv oluşturuldu: $(VENV)"

# ==== Setup ====
.PHONY: install
install: venv
	$(PIP) install -r requirements-rocm.txt
	$(PIP) install -r requirements.txt
	@echo "✔ Bağımlılıklar venv içine kuruldu."

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
.PHONY: gradio
gradio:
	PYTHONPATH=. $(PY) gradio_app/app.py
