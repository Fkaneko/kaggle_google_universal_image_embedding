import torch
from torch.nn import functional as F


class ReconstructionLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        image_embeds_original: torch.Tensor,
        text_embeds_original: torch.Tensor,
        image_embeds_new: torch.Tensor,
        text_embeds_new: torch.Tensor,
    ) -> torch.Tensor:
        cos_similarity_original = image_embeds_original @ text_embeds_original.T
        cos_similarity_new = image_embeds_new @ text_embeds_new.T
        loss = F.mse_loss(
            input=cos_similarity_new, target=cos_similarity_original, reduction="none"
        )
        loss_per_batch = (loss.mean(dim=0) + loss.mean(dim=1)) / 2.0
        return loss_per_batch.mean()
