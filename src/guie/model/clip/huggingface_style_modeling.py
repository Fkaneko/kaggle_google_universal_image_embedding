import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import open_clip
import torch
import torch.nn as nn

# from open_clip.loss import ClipLoss
from torch.nn import functional as F
from transformers.models.clip.modeling_clip import CLIPOutput

logger = logging.getLogger(__name__)


@dataclass
class LossInput:

    image_embeds: torch.Tensor = field(metadata={"help": "CLIP original image embedding"})
    text_embeds: torch.Tensor = field(metadata={"help": "CLIP original text embedding"})
    logit_scale_exp: torch.Tensor = field(metadata={"help": "CLIP exp(logit scale)"})

    image_embeds_compressed: Optional[torch.Tensor] = field(
        default=None,
        metadata={
            "help": "Compressed/Reduced dim CLIP image embedding  ex) middle output of AutoEncoder"
        },
    )
    text_embeds_compressed: Optional[torch.Tensor] = field(
        default=None,
        metadata={
            "help": "Compressed/Reduced dim CLIP text embedding  ex) middle output of AutoEncoder"
        },
    )

    image_embeds_reconstructed: Optional[torch.Tensor] = field(
        default=None,
        metadata={"help": "Reconstructed CLIP image embedding ex) output of AutoEncoder"},
    )
    text_embeds_reconstructed: Optional[torch.Tensor] = field(
        default=None,
        metadata={"help": "Reconstructed CLIP text embedding ex) output of AutoEncoder"},
    )


class ReconstructionLoss(nn.Module):
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
        return (loss.mean(dim=0) + loss.mean(dim=1)) / 2.0


class ClipForCompressedEmbed(nn.Module):
    def __init__(
        self,
        image_head: nn.Module,
        text_head: nn.Module,
        loss_fn: nn.Module,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_e16",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.clip = open_clip.create_model(
            model_name=model_name, pretrained=pretrained, device=device
        )
        self.loss_fn = loss_fn
        self.image_head = image_head
        self.text_head = text_head

    def freeze_clip(self) -> None:
        logger.info("freeze clip pretrained weight")
        for param in self.clip.parameters():
            param.requires_grad = False

    @staticmethod
    def parse_head_output(
        head_output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], embed_dim: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # for jit, avoid using dataclasses for nn.Module output
        # so manually check the content of the output of head_output
        if isinstance(head_output, tuple):
            if len(head_output) == 2:
                embeds_compressed, embeds_reconstructed = head_output
            else:
                raise RuntimeError(f"head_output is too long, {len(head_output)}")
        elif isinstance(head_output, torch.Tensor):
            embeds_compressed = head_output
            embeds_reconstructed = None

        # normalize for cos similarity
        embeds_compressed = F.normalize(embeds_compressed, dim=-1)
        embeds_reconstructed = (
            F.normalize(embeds_reconstructed, dim=-1) if embeds_reconstructed else None
        )

        # check parse result
        assert embeds_compressed.shape[-1] < embed_dim
        if embeds_reconstructed is not None:
            assert embeds_reconstructed.shape[-1] == embed_dim

        return embeds_compressed, embeds_reconstructed

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        # adopted from https://github.com/huggingface/transformers/blob/e54a1b49aa6268c484625c6374f952f318914743/src/transformers/models/clip/modeling_clip.py#L1055-L1074
        image_embeds, text_embeds, logit_scale_exp = self.clip(image=pixel_values, text=input_ids)

        # head
        image_head_output = self.image_head(image_embeds)
        text_head_output = self.text_head(text_embeds)

        # parse head output
        embed_dim = image_embeds.shape[-1]
        image_embeds_compressed, image_embeds_reconstructed = self.parse_head_output(
            head_output=image_head_output, embed_dim=embed_dim
        )
        text_embeds_compressed, text_embeds_reconstructed = self.parse_head_output(
            head_output=text_head_output, embed_dim=embed_dim
        )

        loss = None
        if return_loss:
            loss_input = LossInput(
                image_embeds=image_embeds,
                text_embeds=text_embeds,
                image_embeds_compressed=image_embeds_compressed,
                text_embeds_compressed=text_embeds_compressed,
                image_embeds_reconstructed=image_embeds_reconstructed,
                text_embeds_reconstructed=text_embeds_reconstructed,
                logit_scale_exp=logit_scale_exp,
            )
            # loss = self.loss_fn(
            #     image_features=image_embeds, text_features=text_embeds, logit_scale=logit_scale_exp
            # )
            loss = self.loss_fn(
                image_embeds_original=loss_input.image_embeds,
                text_embeds_original=loss_input.text_embeds,
                image_embeds_new=loss_input.image_embeds_compressed,
                text_embeds_new=loss_input.text_embeds_compressed,
            )
        # we do not use these outputs
        text_outputs, vision_outputs = None, None
        logits_per_text, logits_per_image = None, None

        if not return_dict:
            output = (
                logits_per_image,
                logits_per_text,
                text_embeds,
                image_embeds,
                text_outputs,
                vision_outputs,
            )
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


if __name__ == "__main__":
    from constant import CLIP_INPUT_SIZE, CLIP_MAX_LEN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_head = nn.Sequential(nn.Linear(512, 64, device=device))
    model = ClipForCompressedEmbed(
        image_head=image_head, text_head=image_head, loss_fn=ReconstructionLoss(), device=device
    )

    # prepare input
    # image preprocess for clip model
    image = torch.randn(
        (3, CLIP_INPUT_SIZE[0], CLIP_INPUT_SIZE[1]), dtype=torch.float32, device=device
    )
    # tokenization
    text = torch.randint(low=0, high=1000, size=(CLIP_MAX_LEN,), dtype=torch.int64, device=device)

    model.freeze_clip()
    out = model(
        pixel_values=image.unsqueeze(0),
        input_ids=text.unsqueeze(0),
        return_loss=True,
        return_dict=True,
    )
    print(out)
