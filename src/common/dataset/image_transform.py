import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as FV
from PIL import Image
from torchvision.transforms import (
    AutoAugment,
    CenterCrop,
    ColorJitter,
    Compose,
    ConvertImageDtype,
    InterpolationMode,
    Normalize,
    RandAugment,
    RandomErasing,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToPILImage,
)
from torchvision.transforms.autoaugment import _apply_op

IN_1K_IMG_MEAN = (0.485, 0.456, 0.406)
IN_1K_IMG_STD = (0.229, 0.224, 0.225)


class RandAugmentCustom(RandAugment):
    """Custom torchvision RandAugment  with timm proba behavior"""

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 900,
        num_magnitude_bins: int = 1001,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        std: float = 50.0,
        prob: float = 0.5,
    ) -> None:
        super().__init__(
            num_ops=num_ops,
            magnitude=magnitude,
            num_magnitude_bins=num_magnitude_bins,
            interpolation=interpolation,
            fill=fill,
        )
        self.std = std
        self.prob = prob

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * FV.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        for _ in range(self.num_ops):
            # from https://github.com/rwightman/pytorch-image-models/blob/a520da9b495422bc773fb5dfe10819acb8bd7c5c/timm/data/auto_augment.py#L344
            if self.prob < 1.0 and (torch.rand(1).item() > self.prob):
                continue

            op_meta = self._augmentation_space(self.num_magnitude_bins, FV.get_image_size(img))
            _ = op_meta.pop("Solarize")
            _ = op_meta.pop("Identity")
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            # from timm https://github.com/rwightman/pytorch-image-models/blob/a520da9b495422bc773fb5dfe10819acb8bd7c5c/timm/data/auto_augment.py#L352
            magnitude_with_rand = int(
                torch.normal(torch.tensor(self.magnitude).float(), std=self.std).item()
            )
            magnitude_with_rand = max(0, min(magnitude_with_rand, self.num_magnitude_bins - 1))
            # magnitude_with_rand = min(magnitude_with_rand,
            magnitude = (
                float(magnitudes[magnitude_with_rand].item()) if magnitudes.ndim > 0 else 0.0
            )
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img


# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class TrainTransform(torch.nn.Module):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        crop_scale_range: Tuple[float, float] = (0.3, 1.0),
        rand_augment_magnitude: int = 600,
        color_jitter: float = 0.4,
    ) -> None:
        super().__init__()
        color_jitter = [color_jitter] * 3
        self.transforms = torch.nn.Sequential(
            RandomResizedCrop(
                image_size, scale=crop_scale_range, interpolation=InterpolationMode.BICUBIC
            ),
            # ColorJitter(*color_jitter),
            RandAugmentCustom(
                num_ops=2,
                magnitude=rand_augment_magnitude,
                num_magnitude_bins=1001,
                std=50.0,
                interpolation=InterpolationMode.BILINEAR,
            ),
            RandomHorizontalFlip(p=0.5),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
            RandomErasing(p=0.25, value="random"),
        )

        # CenterCrop(image_size),

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x


class ValTransform(torch.nn.Module):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        crop_pct: float = 0.875,
    ) -> None:
        super().__init__()

        if isinstance(image_size, (tuple, list)):
            scale_size = [int(x / crop_pct) for x in image_size]
        else:
            scale_size = [int(math.floor(image_size / crop_pct))] * 2

        self.transforms = torch.nn.Sequential(
            Resize(scale_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x


class InverseTransform(torch.nn.Module):
    def __init__(
        self,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        mean_np = np.array(mean)
        std_np = np.array(std)
        # at this transforms just for debugging so we do not need nn.Sequential/jit script combination
        self.transforms = Compose(
            [
                Normalize(-np.array(mean_np / std_np), 1.0 / std_np),
                ConvertImageDtype(torch.uint8),
                ToPILImage(),
            ]
        )

    def forward(self, x: torch.Tensor) -> Image:
        x = self.transforms(x)
        return x


if __name__ == "__main__":
    image_size = (224, 224)
    batch_size = 8
    num_channels = 3
    pixel_values = torch.randint(
        0, 255, (batch_size, num_channels, image_size[0], image_size[1]), dtype=torch.uint8
    )
    this_split_transform = TrainTransform(
        image_size=image_size, mean=IN_1K_IMG_MEAN, std=IN_1K_IMG_STD
    )
    this_split_transform = torch.jit.script(this_split_transform)
    out = this_split_transform(pixel_values)
    print(out.shape)
    this_split_transform = ValTransform(
        image_size=image_size, mean=IN_1K_IMG_MEAN, std=IN_1K_IMG_STD
    )
    this_split_transform = torch.jit.script(this_split_transform)
    pixel_values = torch.randint(
        0, 255, (batch_size, num_channels, image_size[0], image_size[1]), dtype=torch.uint8
    )
    out = this_split_transform(pixel_values)
    print(out.shape)
