from typing import List, Optional, Tuple, Union

import numpy as np
import open_clip
import torch
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    ConvertImageDtype,
    InterpolationMode,
    Normalize,
    RandomResizedCrop,
    ToPILImage,
)


# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class Transform(torch.nn.Module):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.transforms = torch.nn.Sequential(
            RandomResizedCrop(
                image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

        # CenterCrop(image_size),

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


def tokenize(texts: Union[str, List[str]]) -> torch.LongTensor:
    # return one dim when single texts
    return open_clip.tokenize(texts=texts).squeeze(0)


if __name__ == "__main__":

    pass
