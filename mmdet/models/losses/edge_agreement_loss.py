import torch
import torch.nn as nn

from mmdet.core import bbox_overlaps
from ..registry import LOSSES
from .utils import weighted_loss
import numpy as np
import torch.nn.functional as F


@LOSSES.register_module
class EdgeAgreementLoss(nn.Module):
    __AVAILABLE_KERNELS = {
        # sobel's kernels
        'sobel-x': np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]]
                            ).reshape((1, 1, 3, 3)),
        'sobel-y': np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]]
                            ).reshape((1, 1, 3, 3)),
        # prewitt's kernels
        'prewitt-x': np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]]
                              ).reshape((1, 1, 3, 3)),
        'prewitt-y': np.array([[1, 1, 1],
                               [0, 0, 0],
                               [-1, -1, -1]]
                              ).reshape((1, 1, 3, 3)),
        # kayyali's kernels
        'kayyali-senw': np.array([[6, 0, -6],
                                  [0, 0, -0],
                                  [-6, 0, 6]]
                                 ).reshape((1, 1, 3, 3)),
        'kayyali-nesw': np.array([[-6, 0, 6],
                                  [0, 0, 0],
                                  [6, 0, -6]]
                                 ).reshape((1, 1, 3, 3)),
        # robert's kernels
        'roberts-x': np.array([[1, 0],
                               [0, -1]]
                              ).reshape((1, 1, 2, 2)),
        'roberts-y': np.array([[0, -1],
                               [1, 0]]
                              ).reshape((1, 1, 2, 2)),
        # laplace's kernel
        'laplacian': np.array([[1, 1, 1],
                               [1, -8, 1],
                               [1, 1, 1]]
                              ).reshape((1, 1, 3, 3))

    }

    __GAUSSIAN_KERNEL = np.array([[0.077847, 0.123317, 0.077847],
                                  [0.123317, 0.195346, 0.1233179],
                                  [0.077847, 0.123317, 0.077847]]
                                 ).reshape((1, 1, 3, 3))

    def __init__(self, filter_names, exponent=2, smooth_target=True,
                 smooth_pred=False, reduction='mean', loss_weight=1.0):
        """Edge Agreement Loss.

        Enforces that the edges of predicted masks match the edges of the target
        masks. This increases the training speed.

        Based on https://doi.org/10.1016/j.cviu.2019.102795

        Args:
            filter_names (list): Name of edge detectors to use
            exponent (int): Exponent for the Lp^p norm.
            smooth_target: Smooth target masks before calculating edges.
            smooth_pred: Smooth predicted masks before calculating edges.
            reduction (string): None, 'mean', or 'sum'
            loss_weight (float): Multiplicative weight of the loss.

        Return:
            Tensor: Loss tensor.
        """

        super(EdgeAgreementLoss, self).__init__()

        for filter_name in filter_names:
            assert filter_name in EdgeAgreementLoss.__AVAILABLE_KERNELS, \
                f'Invalid edge detection kernel name: {filter_name}.'

        self._edge_filters = torch.cat(
            [torch.Tensor(EdgeAgreementLoss.__AVAILABLE_KERNELS[x]) for x in
             filter_names], dim=0)

        self._edge_filters = nn.Parameter(
            self._edge_filters,
            requires_grad=False)

        self.reduction = reduction
        self.loss_weight = loss_weight

        self.exponent = exponent

        self.smooth_pred = smooth_pred
        self.smooth_target = smooth_target

        if smooth_target or smooth_pred:
            self._smoothing_kernel = nn.Parameter(
                torch.Tensor(EdgeAgreementLoss.__GAUSSIAN_KERNEL),
                requires_grad=False)

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0

        assert weight.sum() != 0, \
            'Edge Agreement Loss does not support Class Agnostic Mask mode.'

        assert reduction_override in (None, 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        correct_pred = pred[np.arange(len(weight)), weight]

        target = target.unsqueeze(1)
        correct_pred = correct_pred.unsqueeze(1)

        if self.smooth_pred:
            correct_pred = F.conv2d(correct_pred, self._smoothing_kernel,
                                    padding=1)
        if self.smooth_target:
            target = F.conv2d(target, self._smoothing_kernel, padding=1)

        edges_pred = F.conv2d(correct_pred, self._edge_filters, padding=1)
        edges_target = F.conv2d(target, self._edge_filters, padding=1)

        def lp_loss(x, y, p):
            unreduced_values = torch.pow(torch.abs(x - y), p)
            if reduction == 'mean':
                return unreduced_values.mean()
            elif reduction == 'sum':
                return unreduced_values.sum()
        
        loss = self.loss_weight * lp_loss(
            edges_pred,
            edges_target,
            self.exponent)

        return loss
