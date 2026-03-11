"""
check_dataset_leakage.py
------------------------
Detects duplicate images across train / val / test splits using MD5 hashing.

Usage (run from project root):
    python scripts/check_dataset_leakage.py

    # Custom directories:
    python scripts/check_dataset_leakage.py \\
        --train  data/raw/augmented_alzheimer_mri_dataset/train \\
        --val    data/raw/augmented_alzheimer_mri_dataset/val   \\
        --test   data/raw/augmented_alzheimer_mri_dataset/test

Standard library only — no third-party packages except tqdm (already in requirements.txt).
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    # Graceful fallback if tqdm somehow not installed
    def tqdm(iterable, **kwargs):          # type: ignore
        desc = kwargs.get("desc", "")
        items = list(iterable)
        print(f"{desc}: {len(items)} files")
        return items


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_DATA_ROOT = Path("data/raw/augmented_alzheimer_mri_dataset")


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------
def _md5_file(path: Path, chunk_size: int = 65536) -> str:
    """
    Compute the MD5 hash of a file efficiently by reading in fixed-size chunks.
    chunk_size of 64 KB keeps memory usage flat even for large image files.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_directory(directory: Path, label: str) -> Dict[str, Path]:
    """
    Recursively scan a directory for images and return a dict of
    {md5_hash: first_file_path_seen}.

    Duplicate files within the same split are counted but only one path
    is kept (we care about cross-split leakage, not intra-split dupes).
    """
    if not directory.exists():
        print(f"  [!] Directory not found (skipped): {directory}")
        return {}

    image_files = [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    if not image_files:
        print(f"  [!] No images found in: {directory}")
        return {}

    hash_map: Dict[str, Path] = {}
    for path in tqdm(image_files, desc=f"  Hashing {label:12s}", unit="img", ncols=80):
        digest = _md5_file(path)
        hash_map.setdefault(digest, path)   # keep first occurrence

    return hash_map


# ---------------------------------------------------------------------------
# Cross-split duplicate detection
# ---------------------------------------------------------------------------
def find_duplicates(
    hashes_a: Dict[str, Path],
    hashes_b: Dict[str, Path],
) -> Set[str]:
    """Return the set of MD5 hashes that appear in both hash maps."""
    return set(hashes_a.keys()) & set(hashes_b.keys())


def print_duplicate_samples(
    common: Set[str],
    hashes_a: Dict[str, Path],
    hashes_b: Dict[str, Path],
    label_a: str,
    label_b: str,
    n: int = 3,
):
    """Print up to n example duplicate pairs."""
    if not common:
        return
    print(f"\n  Example duplicates ({label_a} ↔ {label_b}):")
    for i, h in enumerate(list(common)[:n]):
        print(f"    [{i+1}] {hashes_a[h]}")
        print(f"        {hashes_b[h]}")


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------
def run_check(train_dir: Path, val_dir: Path, test_dir: Path) -> int:
    """
    Hash all images in each split and compare across splits.

    Returns:
        0 if no leakage detected, 1 if leakage detected.
    """
    print("\n" + "=" * 60)
    print("  Alzheimer MRI Dataset — Leakage Diagnostic")
    print("=" * 60)
    print(f"\n  Train : {train_dir}")
    print(f"  Val   : {val_dir}")
    print(f"  Test  : {test_dir}\n")

    # ── Hash each split ──────────────────────────────────────────────
    train_hashes = hash_directory(train_dir, "Train")
    val_hashes   = hash_directory(val_dir,   "Val")
    test_hashes  = hash_directory(test_dir,  "Test")

    n_train = len(train_hashes)
    n_val   = len(val_hashes)
    n_test  = len(test_hashes)

    # ── Cross-split comparison ───────────────────────────────────────
    tv_dupes = find_duplicates(train_hashes, val_hashes)
    tt_dupes = find_duplicates(train_hashes, test_hashes)
    vt_dupes = find_duplicates(val_hashes,   test_hashes)

    total_leaks = len(tv_dupes) + len(tt_dupes) + len(vt_dupes)

    # ── Report ───────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  Dataset Leakage Report")
    print("-" * 60)
    print(f"\n  Train images      : {n_train:>8,}")
    print(f"  Validation images : {n_val:>8,}")
    print(f"  Test images       : {n_test:>8,}")
    print()
    print(f"  Duplicates  Train ↔ Validation : {len(tv_dupes):>6,}")
    print(f"  Duplicates  Train ↔ Test       : {len(tt_dupes):>6,}")
    print(f"  Duplicates  Validation ↔ Test  : {len(vt_dupes):>6,}")
    print()

    if total_leaks > 0:
        print("  ⚠️  Potential data leakage detected.")
        print_duplicate_samples(tv_dupes, train_hashes, val_hashes,   "Train", "Val")
        print_duplicate_samples(tt_dupes, train_hashes, test_hashes,  "Train", "Test")
        print_duplicate_samples(vt_dupes, val_hashes,   test_hashes,  "Val",   "Test")
        print()
        print(
            "  Recommendation: Use build_dataloaders() with a unified stratified\n"
            "  split so augmented and original images are never separated across splits.\n"
            "  (The project's data/dataset_loader.py already does this — re-run training.)"
        )
        return_code = 1
    else:
        print("  ✓ No duplicate images detected across splits.")
        return_code = 0

    print("\n" + "=" * 60 + "\n")
    return return_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check for duplicate images between dataset splits (data leakage detector).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_DATA_ROOT / "train",
        help="Path to the training split directory.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=DEFAULT_DATA_ROOT / "val",
        help="Path to the validation split directory.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=DEFAULT_DATA_ROOT / "test",
        help="Path to the test split directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exit_code = run_check(
        train_dir = args.train,
        val_dir   = args.val,
        test_dir  = args.test,
    )
    sys.exit(exit_code)
