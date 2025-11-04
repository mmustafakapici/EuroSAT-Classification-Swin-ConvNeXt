#!/usr/bin/env python
from pathlib import Path
import shutil, random, argparse, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/EuroSAT_RGB")
    ap.add_argument("--out-dir",  default="samples")
    ap.add_argument("--k", type=int, default=1, help="Her sınıftan k adet")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    k        = int(args.k)

    if not data_dir.exists():
        print(f"Data dir yok: {data_dir}. Önce 'make download' çalıştırın.")
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    total = 0
    for cls in classes:
        imgs = list(cls.glob("*.jpg")) + list(cls.glob("*.jpeg")) + list(cls.glob("*.png"))
        if not imgs:
            continue
        picks = random.sample(imgs, min(k, len(imgs)))
        for p in picks:
            shutil.copy2(p, out_dir / f"{cls.name}_{p.name}")
            total += 1
    print(f"✅ {len(classes)} sınıftan toplam {total} görsel {out_dir.resolve()} klasörüne kopyalandı.")

if __name__ == "__main__":
    main()

