import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from mmdet.utils import build_from_cfg
import torch


@LOSSES.register_module
class MultiLoss(nn.Module):

    def __init__(self, loss_configs):
        super().__init__()

        self._losses = nn.ModuleList()
        for loss_config in loss_configs:
            loss = build_from_cfg(loss_config, LOSSES)
            self._losses.append(loss)

    def forward(self, pred, target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        loss_value = 0
        for loss in self._losses:
            loss_value += loss(
                pred,
                target,
                weight=weight,
                avg_factor=avg_factor,
                reduction_override=reduction_override
            )

        return loss_value
