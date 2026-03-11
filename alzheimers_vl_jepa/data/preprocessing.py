"""
preprocessing.py
----------------
Image transforms for training and validation/inference.
- Resize to 128x128 (laptop-friendly)
- Normalize with ImageNet stats (works for fine-tuning)
- Light augmentations for training to avoid overfitting
"""

from torchvision import transforms


# -------------------------------------------------------------------
# Constants — keep in sync with dataset_config.yaml
# -------------------------------------------------------------------
IMAGE_SIZE = 128

# ImageNet mean/std works well even for medical image encoders
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# -------------------------------------------------------------------
# Transform builders
# -------------------------------------------------------------------
def get_train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Training transform:
      - Random horizontal flip (MRI left-right symmetry)
      - Small random rotation (±10°)
      - Slight brightness/contrast jitter
      - Resize → Tensor → Normalize
    Memory-light: no large crops or multi-scale ops.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_val_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Validation / test transform:
      - Resize → Tensor → Normalize (deterministic, no augmentation)
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def denormalize(tensor):
    """
    Reverse the normalization for visualization purposes.
    Input: FloatTensor [C, H, W]
    Output: FloatTensor [C, H, W] in [0, 1]
    """
    import torch
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)
