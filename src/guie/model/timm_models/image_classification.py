from typing import Optional, Tuple, Union

import timm
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention


class ImageClassificationWithTimm(nn.Module):
    """adopted from https://github.com/huggingface/transformers/blob/v4.21.3/src/transformers/models/convnext/modeling_convnext.py#L405"""

    def __init__(
        self,
        timm_model_name: str = "convnext_small",
        num_labels: int = 1000,
        problem_type: str = "single_label_classification",
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.problem_type = problem_type

        self.model = timm.create_model(
            timm_model_name,
            pretrained=True,
            # num_classes=0,
        )
        # # Classifier head
        # self.classifier = (
        #     nn.Linear(hidden_sizes[-1], self..num_labels)
        #     if self..num_labels > 0
        #     else nn.Identity()
        # )

        # # Initialize weights and apply final processing
        # self.post_init()

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
        # logits = self.model(
        #     pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict
        # )
        logits = self.model(pixel_values)
        # pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # logits = self.classifier(pooled_output)

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
    pixel_values = torch.randn(
        (batch_size, num_channels, image_size, image_size), dtype=torch.float32
    )

    model = ImageClassificationWithTimm()
    out = model(pixel_values, return_dict=True)
    print(model)
    print(out.loss)
    print(out.logits.shape)
