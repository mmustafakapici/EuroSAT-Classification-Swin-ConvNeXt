import argparse, pathlib, shutil
import kagglehub

def main(dst:str):
    dst_path = pathlib.Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)

    src_root = pathlib.Path(kagglehub.dataset_download("apollo2506/eurosat-dataset"))
    src_rgb = src_root / "EuroSAT"  # RGB-only
    if not src_rgb.exists():
        raise SystemExit(f"EuroSAT RGB folder not found under {src_root}")

    total = 0
    classes = [d for d in src_rgb.iterdir() if d.is_dir()]
    for cls in classes:
        target = dst_path / cls.name
        target.mkdir(parents=True, exist_ok=True)
        for ext in ("*.jpg","*.jpeg","*.png"):
            for img in cls.glob(ext):
                shutil.copy2(img, target / img.name)
                total += 1
    print(f"✅ Copied {total} images to {dst_path.resolve()} (classes: {len(classes)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="data/EuroSAT_RGB", help="Destination folder in this repo")
    args = ap.parse_args()
    main(args.dst)