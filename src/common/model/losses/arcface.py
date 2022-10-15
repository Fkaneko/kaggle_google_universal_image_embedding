import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ArcMarginProduct_subcenter(nn.Module):
    """adopted from https://github.com/haqishen/Google-Landmark-Recognition-2020-3rd-Place-Solution/blob/main/models.py
    add precomputed class center case
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int = 3,
        precomputed_class_centers: Optional[np.ndarray] = None,
    ):
        super().__init__()
        if precomputed_class_centers is not None:
            assert k == 1
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.weight.requires_grad = False
        else:
            self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters(precomputed_class_centers)
        self.k = k
        self.out_features = out_features

    def reset_parameters(self, precomputed_class_centers: Optional[np.ndarray] = None) -> None:
        if precomputed_class_centers is not None:
            self.weight.data = torch.tensor(
                precomputed_class_centers.astype(np.float32), dtype=torch.float32
            )
        else:
            stdv = 1.0 / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


class ArcFace(torch.nn.Module):
    """ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf)

    from https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py
    official implementation
    from https://github.com/deepinsight/insightface/blob/d4d4531a702e22cc7666cda2de6db53f4dc2e4db/recognition/arcface_torch/dataset.py#L163-L165
    labels (N, )
    :"""

    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        if self.margin == 0.0:
            return logits * self.scale

        # logits means cos = embed @ weight
        index = torch.where(labels != -1)[0]
        # choose: index -> target_batch_index and target class with labels[index]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm
            )
        # except target_logit, keep original logit, cos = embed @ weight
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits


if __name__ == "__main__":
    num_classes = 1000
    in_features = 64
    batch_size = 32
    # prepare input
    precomputed_class_centers = np.random.random((num_classes, in_features)).astype(np.float32)
    embed_inputs = torch.randn((batch_size, in_features), dtype=torch.float32)
    inputs = torch.randn((batch_size, num_classes), dtype=torch.float32)
    labels = torch.randint(0, num_classes, size=(batch_size,), dtype=torch.long)
    arc_margin = ArcMarginProduct_subcenter(
        in_features=in_features,
        out_features=num_classes,
        k=1,
        precomputed_class_centers=precomputed_class_centers,
    )
    out = arc_margin(embed_inputs)
    print(list(arc_margin.parameters()))
    print(out.shape)
    arcface = ArcFace(margin=0.3)
    out = arcface(logits=out, labels=labels)
    print(out.shape)
