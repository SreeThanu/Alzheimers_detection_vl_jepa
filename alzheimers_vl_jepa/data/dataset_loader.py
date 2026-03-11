"""
dataset_loader.py
-----------------
PyTorch Dataset class for loading Alzheimer's MRI images.
- Reads images from the folder structure (class name = folder name)
- Converts class labels to descriptive text prompts (vision-language input)
- Applies transforms for training vs validation/test
- Returns (image_tensor, token_ids, label) tuples
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image


# -------------------------------------------------------------------
# Auto-detection: find the real dataset root even when the extraction
# folder name differs from what the config expects.
# -------------------------------------------------------------------

# Ordered list of folder names the Kaggle dataset is commonly extracted to
_CANDIDATE_DATASET_NAMES = [
    "augmented_alzheimer_mri_dataset",  # our default (config)
    "AugmentedAlzheimerDataset",        # one popular Kaggle extraction name
    "augmented-alzheimer-mri-dataset",  # hyphen variant
    "Dataset",
    "dataset",
    "data",
]

# Sub-paths that should exist inside a valid dataset root
_REQUIRED_SPLITS = ["train", "test"]


def resolve_dataset_root(configured_root: str) -> str:
    """
    Return the real dataset root directory that contains train/ and test/ sub-folders.

    Strategy
    --------
    1. If ``configured_root`` already contains train/ → use it as-is.
    2. Otherwise walk up one level and search sibling directories whose
       names match ``_CANDIDATE_DATASET_NAMES`` (case-insensitive).
    3. Also probes ``data/raw/<candidate>`` relative to the *project root*
       (two levels above data/).
    4. As a last resort, do a shallow BFS under ``data/raw/`` for any
       directory that contains both a train/ and test/ sub-folder.

    Args:
        configured_root: Path from config (e.g. "data/raw/augmented_alzheimer_mri_dataset")

    Returns:
        Resolved absolute-ish path to the dataset root.

    Raises:
        FileNotFoundError: If no valid dataset directory can be located.
    """
    cfg_path = Path(configured_root)

    # 1. Happy path — configured directory already has train/
    if (cfg_path / "train").is_dir():
        return str(cfg_path)

    print(
        f"[dataset] '{configured_root}' does not contain a train/ folder. "
        "Searching for the dataset automatically..."
    )

    # 2. Search sibling directories (same parent as configured_root)
    search_dirs = [
        cfg_path.parent,                    # e.g. data/raw/
        cfg_path.parent.parent,             # e.g. data/
        cfg_path.parent.parent / "raw",     # e.g. data/raw/ (from project root)
        Path("data") / "raw",               # relative to CWD (project root)
        Path("data"),
    ]

    # Named candidates in every search dir
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for candidate in _CANDIDATE_DATASET_NAMES:
            candidate_path = search_dir / candidate
            if (candidate_path / "train").is_dir():
                print(f"[dataset] Detected dataset directory: {candidate_path}")
                return str(candidate_path)

    # 3. BFS: find ANY directory with train/ + test/ under data/raw or data/
    for top in [Path("data") / "raw", Path("data"), cfg_path.parent]:
        if not top.is_dir():
            continue
        for entry in top.iterdir():
            if entry.is_dir() and (entry / "train").is_dir() and (entry / "test").is_dir():
                print(f"[dataset] Auto-detected dataset directory: {entry}")
                return str(entry)
            # One level deeper (e.g. data/raw/zip_name/AugmentedAlzheimerDataset/)
            if entry.is_dir():
                for sub in entry.iterdir():
                    if sub.is_dir() and (sub / "train").is_dir() and (sub / "test").is_dir():
                        print(f"[dataset] Auto-detected dataset directory: {sub}")
                        return str(sub)

    # 4. Give up with an actionable error message
    raise FileNotFoundError(
        f"\n\nCould not find the Alzheimer MRI dataset.\n"
        f"Tried searching from: {[str(d) for d in search_dirs]}\n\n"
        "Please place the dataset so one of these paths exists:\n"
        "  data/raw/augmented_alzheimer_mri_dataset/train/\n"
        "  data/raw/AugmentedAlzheimerDataset/train/\n"
        "  data/raw/Dataset/train/\n\n"
        "Download with: bash scripts/download_dataset.sh\n"
        "Or manually from: https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset"
    )

# -------------------------------------------------------------------
# Default class-to-prompt mapping
# -------------------------------------------------------------------
DEFAULT_CLASS_PROMPTS: Dict[str, str] = {
    "NonDemented":      "This MRI shows no signs of dementia. The brain appears normal.",
    "VeryMildDemented": "This MRI shows very mild dementia with minimal cognitive decline.",
    "MildDemented":     "This MRI shows mild dementia with noticeable cognitive impairment.",
    "ModerateDemented": "This MRI shows moderate dementia with significant brain atrophy.",
}

# Ordered list so label indices are deterministic
CLASS_NAMES: List[str] = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# -------------------------------------------------------------------
# Simple whitespace tokenizer for text prompts
# -------------------------------------------------------------------
class SimpleTokenizer:
    """
    Builds a small fixed vocabulary from the class prompts,
    then converts prompts to integer token sequences.
    No external NLP library required.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, prompts: Dict[str, str], max_seq_len: int = 16):
        self.max_seq_len = max_seq_len
        # Build vocabulary from all words in all prompts
        words = set()
        for prompt in prompts.values():
            words.update(prompt.lower().split())
        self.vocab = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        for word in sorted(words):
            self.vocab[word] = len(self.vocab)
        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> torch.Tensor:
        """Return a fixed-length integer tensor for a given string."""
        tokens = [
            self.vocab.get(word.lower(), self.vocab[self.UNK_TOKEN])
            for word in text.split()
        ]
        # Truncate or pad to max_seq_len
        tokens = tokens[: self.max_seq_len]
        tokens += [self.vocab[self.PAD_TOKEN]] * (self.max_seq_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
class AlzheimerMRIDataset(Dataset):
    """
    Loads MRI images from a directory organised as:
        root/
          <ClassName>/
            *.jpg | *.png | *.jpeg

    Each sample returns:
        image   : FloatTensor  [C, H, W]
        tokens  : LongTensor   [max_seq_len]
        label   : int
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        class_prompts: Optional[Dict[str, str]] = None,
        max_seq_len: int = 16,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_prompts = class_prompts or DEFAULT_CLASS_PROMPTS

        # Build tokenizer from prompts
        self.tokenizer = SimpleTokenizer(self.class_prompts, max_seq_len=max_seq_len)

        # Discover all image paths and their labels
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        """Walk root_dir and collect (image_path, label_idx) pairs."""
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset split directory not found: {self.root_dir}\n"
                "Tip: run resolve_dataset_root() to auto-detect the dataset location."
            )

        found_any_class = False
        for class_name in CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                continue  # Skip missing classes gracefully
            found_any_class = True
            label = CLASS_TO_IDX[class_name]
            for file in class_dir.iterdir():
                if file.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((file, label))

        if not found_any_class:
            raise RuntimeError(
                f"No class subdirectories found under {self.root_dir}.\n"
                f"Expected subdirectory names: {CLASS_NAMES}"
            )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Class directories were found under {self.root_dir} but contained "
                "no image files (jpg/png/bmp).  Did the download complete fully?"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Load image (convert to RGB to handle grayscale MRI files)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Encode text prompt for this class
        class_name = CLASS_NAMES[label]
        prompt = self.class_prompts.get(class_name, "")
        tokens = self.tokenizer.encode(prompt)

        return image, tokens, label

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @staticmethod
    def get_class_names() -> List[str]:
        return CLASS_NAMES


# -------------------------------------------------------------------
# Helper: explicit-split mode (used by evaluation notebooks)
# -------------------------------------------------------------------
def _build_dataloaders_explicit(
    train_dir: str,
    test_dir:  str,
    train_transform,
    val_transform,
    batch_size: int = 16,
    val_frac: float = 0.30,
    num_workers: int = 2,
    seed: int = 42,
    class_prompts: Optional[Dict[str, str]] = None,
    max_seq_len: int = 16,
) -> Tuple["DataLoader", "DataLoader", "DataLoader", int]:
    """
    Mode B helper: loads train images from train_dir, test images from test_dir.
    A stratified val_frac of train_dir is held out as the validation set.
    The test_dir images are NEVER mixed into train/val — true holdout semantics.
    """
    from sklearn.model_selection import train_test_split as _tts

    prompts   = class_prompts or DEFAULT_CLASS_PROMPTS
    tokenizer = SimpleTokenizer(prompts, max_seq_len=max_seq_len)
    VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

    def _collect(directory: str) -> Tuple[List[Path], List[int]]:
        """Walk directory/ClassName/*.img and return (paths, labels)."""
        paths, labels = [], []
        p = Path(directory).resolve()
        if not p.is_dir():
            raise FileNotFoundError(
                f"Directory not found: {directory}\n"
                "Check that the path is correct."
            )
        for class_name in CLASS_NAMES:
            class_dir = p / class_name
            if not class_dir.is_dir():
                continue
            label = CLASS_TO_IDX[class_name]
            for f in class_dir.iterdir():
                if f.suffix.lower() in VALID_EXT:
                    paths.append(f)
                    labels.append(label)
        if not paths:
            raise RuntimeError(
                f"No images found under {directory}\n"
                f"Expected sub-folders named: {CLASS_NAMES}"
            )
        return paths, labels

    # ── Collect all train images, split into train + val ─────────────
    train_paths, train_labels = _collect(train_dir)
    idx_all = list(range(len(train_paths)))
    idx_train, idx_val, _, _ = _tts(
        idx_all, train_labels,
        test_size=val_frac,
        stratify=train_labels,
        random_state=seed,
    )

    # ── Collect test images (true holdout) ───────────────────────────
    test_paths, test_labels = _collect(test_dir)
    idx_test = list(range(len(test_paths)))

    print(f"[dataset] Explicit-split mode:")
    print(f"          train_dir : {train_dir}  ({len(idx_train)} train | {len(idx_val)} val)")
    print(f"          test_dir  : {test_dir}   ({len(idx_test)} test)")
    for i, name in enumerate(CLASS_NAMES):
        count = test_labels.count(i)
        print(f"          {name:25s}: {count:5d} test images")

    # ── Build Dataset objects ─────────────────────────────────────────
    train_ds = _IndexedImageDataset(
        train_paths, train_labels, idx_train,
        transform=train_transform, tokenizer=tokenizer, prompts=prompts,
    )
    val_ds = _IndexedImageDataset(
        train_paths, train_labels, idx_val,
        transform=val_transform, tokenizer=tokenizer, prompts=prompts,
    )
    test_ds = _IndexedImageDataset(
        test_paths, test_labels, idx_test,
        transform=val_transform, tokenizer=tokenizer, prompts=prompts,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    return train_loader, val_loader, test_loader, tokenizer.vocab_size


# -------------------------------------------------------------------
# DataLoader factory  (leakage-free, stratified split)
# -------------------------------------------------------------------
def build_dataloaders(
    data_root: str = "",
    train_transform = None,
    val_transform = None,
    batch_size: int = 16,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    num_workers: int = 2,
    seed: int = 42,
    class_prompts: Optional[Dict[str, str]] = None,
    max_seq_len: int = 16,
    # ── Alternative API: explicit split directories ───────────────
    train_dir: Optional[str] = None,
    test_dir:  Optional[str] = None,
) -> Tuple["DataLoader", "DataLoader", "DataLoader", int]:
    """
    Build leakage-free train / val / test DataLoaders.

    Two calling conventions are supported:

    **A) Single-root mode** (default — 70/15/15 auto-split):
        build_dataloaders(data_root='data/raw/train', ...)
        All images under data_root are split 70/15/15 by stratified sampling.

    **B) Explicit-split mode** (for evaluation against the true holdout):
        build_dataloaders(train_dir='data/raw/train',
                          test_dir='data/raw/test', ...)
        Images from train_dir are split 70% train / 30% val.
        Images from test_dir form the test set (no leakage).

    Args:
        data_root:        Root that contains the dataset images.
                          Used in single-root mode (mode A).
        train_transform:  Torchvision transform for training samples.
        val_transform:    Torchvision transform for val / test samples.
        batch_size:       Mini-batch size.
        train_frac:       Fraction of training data (mode A, default 0.70).
        val_frac:         Fraction for validation   (mode A, default 0.15).
        num_workers:      DataLoader worker processes.
        seed:             Random seed for reproducibility.
        class_prompts:    Optional dict overriding default text prompts.
        max_seq_len:      Token sequence length for the text encoder.
        train_dir:        Explicit train directory (activates mode B).
        test_dir:         Explicit test directory  (activates mode B).

    Returns:
        (train_loader, val_loader, test_loader, vocab_size)
    """
    from sklearn.model_selection import train_test_split as _tts

    # ── Mode B: explicit train_dir + test_dir ────────────────────────
    if train_dir is not None and test_dir is not None:
        return _build_dataloaders_explicit(
            train_dir=train_dir,
            test_dir=test_dir,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=batch_size,
            val_frac=1.0 - train_frac,   # e.g. 0.30 of train_dir → val
            num_workers=num_workers,
            seed=seed,
            class_prompts=class_prompts,
            max_seq_len=max_seq_len,
        )

    prompts   = class_prompts or DEFAULT_CLASS_PROMPTS
    tokenizer = SimpleTokenizer(prompts, max_seq_len=max_seq_len)

    # ── 1. Discover all dataset directories ──────────────────────────
    # Known sub-folder names produced by this Kaggle dataset:
    _KNOWN_DIRS = [
        "AugmentedAlzheimerDataset",
        "OriginalDataset",
        "augmented_alzheimer_mri_dataset",
        "augmented-alzheimer-mri-dataset",
        "Dataset",
        "dataset",
    ]

    data_root_path = Path(data_root).resolve()
    dataset_dirs: List[Path] = []

    # Collect every sub-directory that has at least one class folder inside
    for entry in sorted(data_root_path.iterdir()):
        if not entry.is_dir():
            continue
        # Check whether entry directly contains class folders
        has_classes = any((entry / c).is_dir() for c in CLASS_NAMES)
        # Or entry contains train/ sub-folder
        has_train_split = (entry / "train").is_dir()
        if has_classes or has_train_split:
            dataset_dirs.append(entry)

    if not dataset_dirs:
        # Check if data_root itself contains class folders
        has_classes = any((data_root_path / c).is_dir() for c in CLASS_NAMES)
        if has_classes:
            dataset_dirs = [data_root_path]
        else:
            # Fall back: try auto-detecting a single root with train/
            try:
                resolved = resolve_dataset_root(str(data_root_path / "augmented_alzheimer_mri_dataset"))
                dataset_dirs = [Path(resolved)]
            except FileNotFoundError:
                pass

    if not dataset_dirs:
        raise FileNotFoundError(
            f"No dataset directories found under '{data_root}'.\n"
            "Expected sub-folders like AugmentedAlzheimerDataset/ or OriginalDataset/\n"
            "Download with: bash scripts/download_dataset.sh"
        )

    print(f"[dataset] Combining {len(dataset_dirs)} source director{'y' if len(dataset_dirs)==1 else 'ies'}:")
    for d in dataset_dirs:
        print(f"          • {d}")

    VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

    # ── 2. Collect ALL (image_path, label) pairs ─────────────────────
    all_paths:  List[Path] = []
    all_labels: List[int]  = []

    for ds_dir in dataset_dirs:
        # Support both flat layout (ds_dir/ClassName/*.jpg)
        # and split layout (ds_dir/train/ClassName/*.jpg + ds_dir/test/ClassName/*.jpg)
        sub_dirs: List[Path] = []
        if (ds_dir / "train").is_dir():
            sub_dirs.extend([ds_dir / "train", ds_dir / "test"])
        else:
            sub_dirs.append(ds_dir)

        for sub in sub_dirs:
            for class_name in CLASS_NAMES:
                class_dir = sub / class_name
                if not class_dir.is_dir():
                    continue
                label = CLASS_TO_IDX[class_name]
                for f in class_dir.iterdir():
                    if f.suffix.lower() in VALID_EXT:
                        all_paths.append(f)
                        all_labels.append(label)

    if len(all_paths) == 0:
        raise RuntimeError(
            f"No images found across directories: {dataset_dirs}\n"
            "Check that the dataset downloaded correctly."
        )

    print(f"[dataset] Total images collected : {len(all_paths)}")
    for i, name in enumerate(CLASS_NAMES):
        count = all_labels.count(i)
        print(f"          {name:25s}: {count:6d}  ({100*count/len(all_labels):.1f}%)")

    # ── 3. Stratified 70 / 15 / 15 split ────────────────────────────
    test_frac = 1.0 - train_frac - val_frac   # e.g. 0.15

    # First split: train vs (val + test)
    idx_all = list(range(len(all_paths)))
    idx_train, idx_temp, lbl_train, lbl_temp = _tts(
        idx_all, all_labels,
        test_size=(1.0 - train_frac),
        stratify=all_labels,
        random_state=seed,
    )

    # Second split: val vs test (from the temp pool)
    # val_frac as a fraction of the temp pool
    val_frac_of_temp = val_frac / (val_frac + test_frac)
    idx_val, idx_test, _, _ = _tts(
        idx_temp, lbl_temp,
        test_size=(1.0 - val_frac_of_temp),
        stratify=lbl_temp,
        random_state=seed,
    )

    print(
        f"\n[dataset] Stratified split  →  "
        f"Train: {len(idx_train):6d}  |  "
        f"Val: {len(idx_val):6d}  |  "
        f"Test: {len(idx_test):6d}\n"
    )

    # ── 4. Build Dataset objects for each split ───────────────────────
    train_ds = _IndexedImageDataset(
        all_paths, all_labels, idx_train,
        transform=train_transform, tokenizer=tokenizer, prompts=prompts,
    )
    val_ds = _IndexedImageDataset(
        all_paths, all_labels, idx_val,
        transform=val_transform, tokenizer=tokenizer, prompts=prompts,
    )
    test_ds = _IndexedImageDataset(
        all_paths, all_labels, idx_test,
        transform=val_transform, tokenizer=tokenizer, prompts=prompts,
    )

    # ── 5. Build DataLoaders ─────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    return train_loader, val_loader, test_loader, tokenizer.vocab_size


# -------------------------------------------------------------------
# Indexed dataset that applies a transform to arbitrary index subsets
# -------------------------------------------------------------------
class _IndexedImageDataset(Dataset):
    """
    Lightweight dataset backed by pre-computed (path, label) lists and
    an explicit index subset (from the stratified split).
    Applies the given transform on-the-fly when images are loaded.
    """

    def __init__(
        self,
        all_paths: List[Path],
        all_labels: List[int],
        indices: List[int],
        transform: Optional[Callable],
        tokenizer: "SimpleTokenizer",
        prompts: Dict[str, str],
    ):
        self.paths     = all_paths
        self.labels    = all_labels
        self.indices   = indices
        self.transform = transform
        self.tokenizer = tokenizer
        self.prompts   = prompts

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx  = self.indices[idx]
        img_path  = self.paths[real_idx]
        label     = self.labels[real_idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        class_name = CLASS_NAMES[label]
        prompt     = self.prompts.get(class_name, "")
        tokens     = self.tokenizer.encode(prompt)

        return image, tokens, label


# -------------------------------------------------------------------
# Legacy helper kept for backward-compatibility with notebooks /
# _TransformOverrideSubset usage — no functional role in new pipeline
# -------------------------------------------------------------------
class _TransformOverrideSubset(Dataset):
    """Wraps a torch Subset and applies a different transform."""

    def __init__(self, subset: torch.utils.data.Subset, transform: Callable):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        inner: AlzheimerMRIDataset = self.subset.dataset  # type: ignore
        real_idx = self.subset.indices[idx]
        img_path, label = inner.samples[real_idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        class_name = CLASS_NAMES[label]
        prompt = inner.class_prompts.get(class_name, "")
        tokens = inner.tokenizer.encode(prompt)

        return image, tokens, label
