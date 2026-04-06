#!/usr/bin/env python3
"""
============================================================
OSCC Detection Project — Dataset Downloader (FIXED v2)
============================================================
Compatible with kaggle package version 1.7.x+

FIX: Uses kaggle CLI executable directly from venv Scripts
     folder, avoiding the broken Python API import.
============================================================
"""

# ── standard library ──────────────────────────────────────
import os
import sys
import json
import shutil
import random
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List

# ── set global random seed ────────────────────────────────
SEED: int = 42
random.seed(SEED)

# ── project paths ─────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"

# Kaggle dataset identifier
KAGGLE_DATASET: str = "ashenafifasilkebede/dataset"

# Split ratios
TRAIN_RATIO: float = 0.70
VAL_RATIO: float   = 0.15
TEST_RATIO: float  = 0.15

# Supported image extensions
IMAGE_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ────────────────────────────────────────────────────────────
# 1. ENSURE KAGGLE IS INSTALLED
# ────────────────────────────────────────────────────────────
def ensure_kaggle_installed() -> None:
    """Check kaggle is installed, install if missing."""
    try:
        import kaggle  # noqa: F401
        print("✅ Kaggle package is already installed.")
    except ImportError:
        print("⏳ Installing kaggle …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("✅ Kaggle installed.")


# ────────────────────────────────────────────────────────────
# 2. VALIDATE KAGGLE CREDENTIALS
# ────────────────────────────────────────────────────────────
def check_kaggle_credentials() -> bool:
    """Verify ~/.kaggle/kaggle.json exists and is valid."""
    kaggle_json: Path = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_json.exists():
        print("❌ kaggle.json not found at:", kaggle_json)
        print("   Go to https://www.kaggle.com/account → API → Create New Token")
        return False

    try:
        with open(kaggle_json, "r") as f:
            creds = json.load(f)
        if "username" not in creds or "key" not in creds:
            print("❌ kaggle.json missing 'username' or 'key'.")
            return False
        print(f"✅ Kaggle credentials found for user: {creds['username']}")
        return True
    except Exception as exc:
        print(f"❌ Error reading kaggle.json: {exc}")
        return False


# ────────────────────────────────────────────────────────────
# 3. FIND KAGGLE EXECUTABLE
# ────────────────────────────────────────────────────────────
def find_kaggle_exe() -> str:
    """
    Find kaggle executable inside the current venv Scripts folder.
    Returns the path as string.
    """
    python_exe = Path(sys.executable)
    scripts_dir = python_exe.parent  # e.g. oscc_env/Scripts/

    # Windows
    kaggle_win = scripts_dir / "kaggle.exe"
    if kaggle_win.exists():
        return str(kaggle_win)

    # Linux / Mac
    kaggle_unix = scripts_dir / "kaggle"
    if kaggle_unix.exists():
        return str(kaggle_unix)

    # Fallback: rely on system PATH
    return "kaggle"


# ────────────────────────────────────────────────────────────
# 4. DOWNLOAD DATASET
# ────────────────────────────────────────────────────────────
def download_dataset() -> None:
    """
    Download dataset using the kaggle CLI executable found
    inside the active virtual environment.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print("✅ Dataset already exists in data/raw/ — skipping download.")
        return

    kaggle_exe = find_kaggle_exe()
    print(f"⏳ Downloading dataset: {KAGGLE_DATASET} …")
    print(f"   Using: {kaggle_exe}")
    print("   This may take several minutes …\n")

    try:
        result = subprocess.run(
            [
                kaggle_exe,
                "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", str(DATA_DIR),
                "--force",
            ],
            text=True,
        )

        if result.returncode != 0:
            print(f"\n❌ Download failed (exit code {result.returncode})")
            _print_manual_instructions()
            sys.exit(1)

        print("\n✅ Download complete.")

    except FileNotFoundError:
        print(f"❌ Could not find kaggle executable at: {kaggle_exe}")
        print("   Run: pip install kaggle  then try again.")
        _print_manual_instructions()
        sys.exit(1)

    # ── Extract zip ───────────────────────────────────────
    _extract_zip()


# ────────────────────────────────────────────────────────────
# 5. EXTRACT ZIP
# ────────────────────────────────────────────────────────────
def _extract_zip() -> None:
    """Find and extract the downloaded zip into data/raw/."""
    zip_file: Path = DATA_DIR / "dataset.zip"
    if not zip_file.exists():
        zips = list(DATA_DIR.glob("*.zip"))
        if not zips:
            print("❌ No zip file found after download.")
            _print_manual_instructions()
            sys.exit(1)
        zip_file = zips[0]

    print(f"⏳ Extracting {zip_file.name} → data/raw/ …")
    try:
        with zipfile.ZipFile(zip_file, "r") as zf:
            print(f"   Files to extract: {len(zf.namelist())}")
            zf.extractall(RAW_DIR)
        print("✅ Extraction complete.")
        zip_file.unlink(missing_ok=True)
        print("🗑️  Removed zip file.")
    except zipfile.BadZipFile as exc:
        print(f"❌ Bad zip file: {exc}")
        sys.exit(1)


# ────────────────────────────────────────────────────────────
# 6. CHECK FOR MANUALLY PLACED ZIP
# ────────────────────────────────────────────────────────────
def check_manual_zip() -> bool:
    """
    If user manually downloaded the zip into data/ folder,
    detect and extract it automatically.
    """
    zips = list(DATA_DIR.glob("*.zip"))
    if not zips:
        return False

    zip_file = zips[0]
    print(f"📦 Found manually placed zip: {zip_file.name}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_file, "r") as zf:
            print(f"   Extracting {len(zf.namelist())} files …")
            zf.extractall(RAW_DIR)
        print("✅ Extraction complete.")
        zip_file.unlink(missing_ok=True)
        return True
    except zipfile.BadZipFile as exc:
        print(f"❌ Bad zip: {exc}")
        return False


# ────────────────────────────────────────────────────────────
# 7. MANUAL DOWNLOAD INSTRUCTIONS
# ────────────────────────────────────────────────────────────
def _print_manual_instructions() -> None:
    """Print step-by-step manual download guide."""
    print("\n" + "=" * 60)
    print("📥  MANUAL DOWNLOAD (Alternative if auto fails)")
    print("=" * 60)
    print("""
  1. Open: https://www.kaggle.com/datasets/ashenafifasilkebede/dataset
  2. Log in to Kaggle
  3. Click the "Download" button
  4. Save the zip to:
       D:\\OSCC_P\\oscc_detection\\data\\
  5. Run this script again — it will auto-detect and extract it

  OR extract the zip manually:
  - Extract into: D:\\OSCC_P\\oscc_detection\\data\\raw\\
  - Folder should contain 'normal' and 'oscc' subfolders
  - Then run this script again to split into train/val/test
""")


# ────────────────────────────────────────────────────────────
# 8. DISCOVER CLASS FOLDERS
# ────────────────────────────────────────────────────────────
def discover_classes(raw_root: Path) -> Dict[str, List[Path]]:
    """Walk raw_root and categorise images into normal/oscc."""
    classes: Dict[str, List[Path]] = {"normal": [], "oscc": []}

    for root, _dirs, files in os.walk(raw_root):
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            parent_name = fpath.parent.name.lower()
            if "normal" in parent_name:
                classes["normal"].append(fpath)
            else:
                classes["oscc"].append(fpath)

    return classes


# ────────────────────────────────────────────────────────────
# 9. SPLIT AND ORGANISE
# ────────────────────────────────────────────────────────────
def split_and_organise(
    classes: Dict[str, List[Path]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float   = VAL_RATIO,
    test_ratio: float  = TEST_RATIO,
) -> Dict[str, Dict[str, int]]:
    """Shuffle and copy images into train/val/test splits."""
    stats: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for cls_name, image_paths in classes.items():
        random.shuffle(image_paths)
        n       = len(image_paths)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        splits = {
            "train": image_paths[:n_train],
            "val":   image_paths[n_train: n_train + n_val],
            "test":  image_paths[n_train + n_val:],
        }

        for split_name, paths in splits.items():
            dest_dir = DATA_DIR / split_name / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for src_path in paths:
                dst_path = dest_dir / src_path.name
                counter = 1
                while dst_path.exists():
                    dst_path = dest_dir / f"{src_path.stem}_{counter}{src_path.suffix}"
                    counter += 1
                shutil.copy2(src_path, dst_path)

            stats[split_name][cls_name] = len(paths)

    return stats


# ────────────────────────────────────────────────────────────
# 10. PRINT STATISTICS
# ────────────────────────────────────────────────────────────
def print_statistics(stats: Dict[str, Dict[str, int]]) -> None:
    """Pretty-print dataset split summary table."""
    print("\n" + "=" * 55)
    print("           📊  DATASET STATISTICS")
    print("=" * 55)
    print(f"{'Split':<10} {'Normal':>10} {'OSCC':>10} {'Total':>10}")
    print("-" * 55)

    grand_total = 0
    for split in ("train", "val", "test"):
        n_normal = stats[split].get("normal", 0)
        n_oscc   = stats[split].get("oscc",   0)
        total    = n_normal + n_oscc
        grand_total += total
        print(f"{split:<10} {n_normal:>10} {n_oscc:>10} {total:>10}")

    print("-" * 55)
    all_normal = sum(stats[s].get("normal", 0) for s in stats)
    all_oscc   = sum(stats[s].get("oscc",   0) for s in stats)
    print(f"{'TOTAL':<10} {all_normal:>10} {all_oscc:>10} {grand_total:>10}")
    print("=" * 55 + "\n")


# ────────────────────────────────────────────────────────────
# 11. CHECK EXISTING SPLITS
# ────────────────────────────────────────────────────────────
def splits_already_exist() -> bool:
    """Return True if all train/val/test splits have images."""
    for split in ("train", "val", "test"):
        for cls in ("normal", "oscc"):
            folder = DATA_DIR / split / cls
            if not folder.exists() or not any(folder.iterdir()):
                return False
    return True


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
def main() -> None:
    """Full pipeline: download → extract → split → stats."""
    print("\n" + "=" * 55)
    print("  🔬  OSCC Dataset Downloader & Organiser")
    print("=" * 55 + "\n")

    # Step 1: Kaggle package check
    ensure_kaggle_installed()

    # Step 2: Credentials check
    if not check_kaggle_credentials():
        sys.exit(1)

    # Step 3: Already split?
    if splits_already_exist():
        print("✅ Dataset splits already exist — skipping download.")
        stats: Dict[str, Dict[str, int]] = {}
        for split in ("train", "val", "test"):
            stats[split] = {}
            for cls in ("normal", "oscc"):
                folder = DATA_DIR / split / cls
                count = len([
                    f for f in folder.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS
                ])
                stats[split][cls] = count
        print_statistics(stats)
        return

    # Step 4: Check for manually placed zip
    raw_empty = not RAW_DIR.exists() or not any(RAW_DIR.iterdir())
    if raw_empty:
        if check_manual_zip():
            print("✅ Manual zip detected and extracted.")
        else:
            # Step 5: Auto-download
            download_dataset()

    # Step 6: Discover images
    print("⏳ Scanning data/raw/ for images …")
    classes = discover_classes(RAW_DIR)

    if not classes["normal"] and not classes["oscc"]:
        print("❌ No images found in data/raw/")
        print("   Raw folder contents:")
        for item in RAW_DIR.rglob("*"):
            if item.is_file():
                print(f"     {item.relative_to(RAW_DIR)}")
        _print_manual_instructions()
        sys.exit(1)

    print(f"   Found {len(classes['normal']):,} Normal images")
    print(f"   Found {len(classes['oscc']):,}   OSCC   images")

    # Step 7: Split
    print("⏳ Splitting 70 / 15 / 15 …")
    stats = split_and_organise(classes)

    # Step 8: Print stats
    print_statistics(stats)
    print("✅ Dataset ready! Next step: python src/preprocess.py\n")


if __name__ == "__main__":
    main()