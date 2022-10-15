import logging
from typing import Optional, Tuple, Union

import numpy as np
import open_clip
import timm
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torchvision.transforms import Normalize
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

from src.common.dataset.image_transform import IN_1K_IMG_MEAN, IN_1K_IMG_STD
from src.common.model.losses.arcface import ArcFace, ArcMarginProduct_subcenter
from src.guie.model.clip.constant import CLIP_INPUT_MEAN, CLIP_INPUT_STD

logger = logging.getLogger(__name__)


class Projection(nn.Module):
    def __init__(
        self,
        in_features: int = 768,
        middle_size: int = 512,
        out_features: int = 64,
        use_two_layer: bool = False,
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(64)
        if use_two_layer:
            self.fc1 = nn.Linear(in_features=in_features, out_features=middle_size)
            self.fc2 = nn.Identity()
            self.norm = nn.LayerNorm(middle_size)
            self.gelu = nn.GELU()
            self.fc3 = nn.Linear(in_features=middle_size, out_features=out_features)
        else:
            fc2_out = int(out_features * 4)
            self.fc1 = nn.Linear(in_features=in_features, out_features=middle_size)
            self.fc2 = nn.Linear(in_features=middle_size, out_features=fc2_out)
            self.norm = nn.LayerNorm(fc2_out)
            self.gelu = nn.GELU()
            self.fc3 = nn.Linear(in_features=fc2_out, out_features=out_features)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.fc1(x)
        baseline = self.avg_pool(x)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.fc3(x)
        x += baseline
        return x


class ArcFaceWithTimm(nn.Module):
    """adopted from https://github.com/huggingface/transformers/blob/v4.21.3/src/transformers/models/convnext/modeling_convnext.py#L405"""

    def __init__(
        self,
        timm_model_name: str = "convnext_small",
        num_labels: int = 1000,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        problem_type: str = "single_label_classification",
        embed_dim: int = 64,
        use_two_layer_on_projection: bool = False,
        num_sub_centers: int = 1,
        arcface_scale: float = 30.0,
        margin: float = 0.3,
        precomputed_class_centers: Optional[np.ndarray] = None,
        is_freeze_backbone: bool = False,
        teacher_clip_model: Optional[str] = None,
        teacher_pretrained: Optional[str] = None,
        class_loss_weight: float = 1.0,
        distill_loss_weight: float = 0.5,
        distill_cos_loss_weight: float = 3.0,
        distill_cos_loss_fn: Optional[nn.Module] = None,
        use_binary_teacher: bool = False,
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.problem_type = problem_type
        self.model = timm.create_model(
            timm_model_name,
            pretrained=True,
            num_classes=0,  # without final fc layer
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        backbone_dim = 768
        teacher_embed_size = 512
        self.projection = Projection(
            in_features=backbone_dim,
            middle_size=teacher_embed_size,
            out_features=embed_dim,
            use_two_layer=use_two_layer_on_projection,
        )

        self.arc_margin_product = ArcMarginProduct_subcenter(
            in_features=embed_dim,
            out_features=self.num_labels,
            k=num_sub_centers,
            precomputed_class_centers=precomputed_class_centers,
        )
        self.arcface = ArcFace(s=arcface_scale, margin=margin)

        if is_freeze_backbone:
            self.freeze_backbone()

        self.class_loss_weight = class_loss_weight
        self.teacher_clip_model = teacher_clip_model
        self.distill_loss_weight = distill_loss_weight
        self.distill_cos_loss_weight = distill_cos_loss_weight
        self.distill_cos_loss_fn = distill_cos_loss_fn
        self.use_binary_teacher = use_binary_teacher
        if teacher_clip_model:
            assert teacher_pretrained is not None
            self.clip = open_clip.create_model(
                model_name=teacher_clip_model, pretrained=teacher_pretrained
            )
            # delete language part
            self.clip.transformer = nn.Identity()
            for param in self.clip.parameters():
                param.requires_grad = False
            self.in1k_to_clip = self.__get_imagenet_norm_to_clip_norm()

        # # Initialize weights and apply final processing
        # self.post_init()

    def freeze_backbone(self) -> None:
        logger.info("freeze backbone pretrained weight")
        for param in self.model.parameters():
            param.requires_grad = False

    def __get_imagenet_norm_to_clip_norm(self) -> nn.Module:
        mean_clip = np.array(CLIP_INPUT_MEAN)
        std_clip = np.array(CLIP_INPUT_STD)

        mean = np.array(IN_1K_IMG_MEAN)
        std = np.array(IN_1K_IMG_STD)

        in1k_to_clip = Normalize(+(mean_clip - mean) / std, std_clip / std)
        return in1k_to_clip

    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        x = self.model(pixel_values)
        student_out = self.projection.fc1(x)
        baseline = self.projection.avg_pool(student_out)
        x = self.projection.fc2(student_out)
        x = self.projection.norm(x)
        x = self.projection.gelu(x)
        x = self.projection.fc3(x)
        x += baseline

        distill_loss, distill_cos_loss = None, None
        if self.teacher_clip_model is not None:
            clip_input = self.in1k_to_clip(pixel_values)
            teacher_out = self.clip.encode_image(clip_input)
            teacher_out = F.normalize(teacher_out, p=2.0, dim=-1)
            distill_loss = F.l1_loss(
                input=F.normalize(student_out, p=2.0, dim=-1),
                target=teacher_out,
            )
            if self.distill_cos_loss_fn is not None:
                if self.use_binary_teacher:
                    teacher_out = F.normalize(teacher_out.sign(), p=2.0, dim=-1)
                student_compressed_out = F.normalize(x, p=2.0, dim=-1)
                distill_cos_loss = self.distill_cos_loss_fn(
                    image_embeds_original=teacher_out,
                    text_embeds_original=teacher_out,
                    image_embeds_new=student_compressed_out,
                    text_embeds_new=student_compressed_out,
                )

        logits = self.arc_margin_product(x)

        # cast to fp32 for arcface calculation
        orig_dtype = x.dtype
        logits = logits.float()
        logits = self.arcface(logits=logits, labels=labels)
        # logits = logits.to(orig_dtype)

        loss = None
        if labels is not None:
            if self.problem_type is None:
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

            if distill_loss is not None:
                loss = self.class_loss_weight * loss + self.distill_loss_weight * distill_loss
            if distill_cos_loss is not None:
                loss = loss + self.distill_cos_loss_weight * distill_cos_loss
        if not return_dict:
            # output = (logits,) + outputs[2:]
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
        )


if __name__ == "__main__":

    image_size = 224
    batch_size = 8
    num_channels = 3
    num_classes = 1000
    teacher_clip_model = "ViT-B-32"
    teacher_pretrained = "laion2b_e16"

    model = ArcFaceWithTimm(
        num_labels=num_classes,
        arcface_scale=30.0,
        margin=0.3,
        teacher_clip_model=teacher_clip_model,
        teacher_pretrained=teacher_pretrained,
    )
    pixel_values = torch.randn(
        (batch_size, num_channels, image_size, image_size), dtype=torch.float32
    )
    labels = torch.randint(0, num_classes, size=(batch_size,), dtype=torch.long)

    out = model(pixel_values, labels=labels, return_dict=True)
    print(out.loss)
    print(out.logits.shape)
    print(out.logits.max(dim=-1))
