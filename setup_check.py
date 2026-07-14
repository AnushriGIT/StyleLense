"""
setup_check.py — Run this first to verify your environment is ready.

What it checks:
  ✅ All required libraries are installed
  ✅ GPU is available (optional but speeds up processing)
  ✅ Dataset directory exists and has images
  ✅ Key annotation files are present

Usage:
  python setup_check.py
"""

import sys
import os

print("=" * 60)
print("  Fashion Assistant — Environment Check")
print("=" * 60)

# ── 1. Check Python version ───────────────────────────────────────────────────
print(f"\n📌 Python version: {sys.version.split()[0]}")
if sys.version_info < (3, 8):
    print("  ❌ Python 3.8+ required. Please upgrade.")
    sys.exit(1)
else:
    print("  ✅ Python version OK")

# ── 2. Check required libraries ───────────────────────────────────────────────
print("\n📦 Checking required libraries...")
libs = {
    "torch": "PyTorch (deep learning)",
    "torchvision": "TorchVision (pre-trained models & image transforms)",
    "PIL": "Pillow (image loading)",
    "cv2": "OpenCV (image processing)",
    "faiss": "FAISS (similarity search)",
    "numpy": "NumPy (numerical computing)",
    "sklearn": "Scikit-learn (ML utilities)",
    "flask": "Flask (web server)",
    "tqdm": "tqdm (progress bars)",
}

all_ok = True
for lib, description in libs.items():
    try:
        __import__(lib)
        print(f"  ✅ {lib:15s} — {description}")
    except ImportError:
        print(f"  ❌ {lib:15s} — {description}  [NOT INSTALLED]")
        all_ok = False

if not all_ok:
    print("\n⚠️  Some libraries are missing. Install them with:")
    print("    pip install -r requirements.txt")
else:
    print("\n  ✅ All libraries installed!")

# ── 3. Check GPU ───────────────────────────────────────────────────────────────
print("\n🖥️  Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  ✅ GPU found: {gpu_name}")
        print(f"     CUDA version: {torch.version.cuda}")
    else:
        print("  ⚠️  No GPU found — will use CPU (slower but works fine)")
        print("     For a quick demo, CPU is totally fine.")
except Exception as e:
    print(f"  ⚠️  Could not check GPU: {e}")

# ── 4. Check dataset ──────────────────────────────────────────────────────────
print("\n📂 Checking dataset...")
try:
    from config import DATASET_BASE_PATH, IMG_DIR, ANNO_DIR, CATEGORY_IMG_FILE

    if os.path.exists(DATASET_BASE_PATH):
        print(f"  ✅ Dataset folder found: {DATASET_BASE_PATH}")
    else:
        print(f"  ❌ Dataset folder NOT found at: {DATASET_BASE_PATH}")
        print("     Update DATASET_BASE_PATH in config.py to your dataset location.")

    if os.path.exists(IMG_DIR):
        # Count images
        count = 0
        for root, dirs, files in os.walk(IMG_DIR):
            count += sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")))
            if count > 1000:  # Stop counting early for large datasets
                print("  ✅ img/ folder found — 1000+ images detected (counting stopped early)")
                break
        else:
            print(f"  ✅ img/ folder found — {count} images detected")
    else:
        print("  ❌ img/ subfolder NOT found inside dataset folder")
        print(f"     Expected: {IMG_DIR}")

    if os.path.exists(ANNO_DIR):
        anno_files = os.listdir(ANNO_DIR)
        print(f"  ✅ Anno/ folder found — {len(anno_files)} files: {anno_files[:5]}")
    else:
        print("  ⚠️  Anno/ folder not found — needed for category labels")
        print(f"     Expected: {ANNO_DIR}")

    if os.path.exists(CATEGORY_IMG_FILE):
        with open(CATEGORY_IMG_FILE) as f:
            lines = f.readlines()
        print(f"  ✅ list_category_img.txt found — {len(lines)} entries")
    else:
        print("  ⚠️  list_category_img.txt not found — optional for basic search")

except ImportError as e:
    print(f"  ❌ Could not import config.py: {e}")

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Setup check complete!")
print("  Next step → run: python step2_dataset_explorer.py")
print("=" * 60)
