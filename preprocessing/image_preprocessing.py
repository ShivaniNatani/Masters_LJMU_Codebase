import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from typing import Tuple, Union

from utils.logger import setup_logger

logger = setup_logger("image_preprocessing")


def get_image_transforms(
    image_size: Tuple[int, int] = (224, 224),
    is_training: bool = False,
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """
    Constructs PyTorch image transformation pipeline for radiology images.
    """
    transform_list = []

    transform_list.append(T.Resize(image_size))

    if is_training:
        transform_list.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
            ]
        )

    transform_list.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )

    return T.Compose(transform_list)


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """
    Reads an image from disk using OpenCV / PIL, resizes, normalizes, and returns a Torch Tensor (3, H, W).
    """
    if not image_path or not isinstance(image_path, str):
        # Return fallback zero tensor for missing images
        logger.warning("preprocess_image: empty/invalid image_path -> returning zero tensor")
        return torch.zeros((3, target_size[0], target_size[1]))

    transform = get_image_transforms(image_size=target_size, is_training=False)

    try:
        # Context manager ensures the file handle is released immediately
        # rather than left open until garbage collection - important under
        # multi-worker DataLoaders issuing many __getitem__ calls per second.
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            return transform(img)
    except Exception as e:
        logger.warning(f"PIL failed to read '{image_path}' ({e}); retrying with OpenCV fallback")
        # Fallback OpenCV reader
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            logger.warning(f"OpenCV fallback also failed for '{image_path}' -> returning zero tensor")
            return torch.zeros((3, target_size[0], target_size[1]))
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        return transform(img_pil)
