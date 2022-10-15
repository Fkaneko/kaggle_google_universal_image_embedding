import copy
import logging
from typing import Tuple

from torch import nn

logger = logging.getLogger(__name__)


class MovingAvg:
    # adopted from https://github.com/salesforce/ensemble-of-averages/blob/main/domainbed/algorithms.py#L118-L171
    def __init__(
        self,
        network: nn.Module,
        sma_start_iter: int = 2,
        update_freq_steps: int = 1,
        on_gpu: bool = True,
        freeze_part_prefix: Tuple[str, ...] = ("model",),
    ):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        if on_gpu:
            self.network_sma.cuda()
        self.sma_start_iter = sma_start_iter
        self.update_freq_steps = update_freq_steps
        self.freeze_part_prefix = freeze_part_prefix

        self.global_iter = 0
        self.sma_count = 0
        self.last_state_dict = {}

    def update_sma(self):
        self.global_iter += 1
        new_dict = {}
        is_strict = True
        if (self.global_iter >= self.sma_start_iter) & (
            self.global_iter % self.update_freq_steps == 0
        ):
            self.sma_count += 1
            for (name, param_q), (_, param_k) in zip(
                self.network.state_dict().items(), self.network_sma.state_dict().items()
            ):
                if name.split(".")[0] in self.freeze_part_prefix:
                    is_strict = False
                    continue

                if "num_batches_tracked" not in name:
                    new_dict[name] = (
                        param_k.data.detach().clone() * self.sma_count
                        + param_q.data.detach().clone()
                    ) / (1.0 + self.sma_count)
        else:
            for (name, param_q), (_, param_k) in zip(
                self.network.state_dict().items(), self.network_sma.state_dict().items()
            ):
                if name.split(".")[0] in self.freeze_part_prefix:
                    is_strict = False
                    continue
                if "num_batches_tracked" not in name:
                    new_dict[name] = param_q.detach().data.clone()

        self.last_state_dict = new_dict
        self.network_sma.load_state_dict(new_dict, strict=is_strict)
