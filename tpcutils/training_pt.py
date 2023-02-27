import warnings
from abc import abstractmethod

import numpy as np
import scipy
import torch
from torch.optim.lr_scheduler import _LRScheduler

from torch.nn.modules.loss import _Loss
from torch.nn.functional import softplus


class PiecewiseLinearLR(_LRScheduler):
    """Interpolate learning rate linearly between milestones."""

    def __init__(
        self,
        optimizer,
        milestones,
        factors,
        last_epoch=-1,
        verbose=False,
    ):
        # Check(s)
        if milestones != sorted(milestones):
            raise ValueError("Milestones must be increasing")
        if len(milestones) != len(factors):
            raise ValueError(
                "Only multiplicative factor must be specified for each milestone."
            )

        self.milestones = milestones
        self.factors = factors
        super().__init__(optimizer, last_epoch, verbose)

    def _get_factor(self):
        # Linearly interpolate multiplicative factor between milestones.
        return np.interp(self.last_epoch, self.milestones, self.factors)

    def get_lr(self):
        """Get effective learning rate(s) for each optimizer."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        return [base_lr * self._get_factor() for base_lr in self.base_lrs]
    


class LogCMK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, m, kappa):
        """Forward pass."""
        dtype = kappa.dtype
        ctx.save_for_backward(kappa)
        ctx.m = m
        ctx.dtype = dtype
        kappa = kappa.double()
        iv = torch.from_numpy(
            scipy.special.iv(m / 2.0 - 1, kappa.cpu().numpy())
        ).to(kappa.device)
        return (
            (m / 2.0 - 1) * torch.log(kappa)
            - torch.log(iv)
            - (m / 2) * np.log(2 * np.pi)
        ).type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass."""
        kappa = ctx.saved_tensors[0]
        m = ctx.m
        dtype = ctx.dtype
        kappa = kappa.double().cpu().numpy()
        grads = -(
            (scipy.special.iv(m / 2.0, kappa))
            / (scipy.special.iv(m / 2.0 - 1, kappa))
        )
        return (
            None,
            grad_output
            * torch.from_numpy(grads).to(grad_output.device).type(dtype),
        )


class VonMisesFisherLoss(_Loss):

    @classmethod
    def log_cmk_exact(
        cls, m, kappa):
        return LogCMK.apply(m, kappa)

    @classmethod
    def log_cmk_approx(cls, m, kappa): 

        v = m / 2.0 - 0.5
        a = torch.sqrt((v + 1) ** 2 + kappa**2)
        b = v - 1
        return -a + b * torch.log(b + a)

    @classmethod
    def log_cmk(cls, m, kappa, kappa_switch = 100.0):

        kappa_switch = torch.tensor([kappa_switch]).to(kappa.device)
        mask_exact = kappa < kappa_switch

        # Ensure continuity at `kappa_switch`
        offset = cls.log_cmk_approx(m, kappa_switch) - cls.log_cmk_exact(
            m, kappa_switch
        )
        ret = cls.log_cmk_approx(m, kappa) - offset
        ret[mask_exact] = cls.log_cmk_exact(m, kappa[mask_exact])
        return ret

    def _evaluate(self, prediction, target):

        # Check(s)
        assert prediction.dim() == 2
        assert target.dim() == 2
        assert prediction.size() == target.size()

        # Computing loss
        m = target.size()[1]
        k = torch.norm(prediction, dim=1)
        dotprod = torch.sum(prediction * target, dim=1)
        elements = -self.log_cmk(m, k) - dotprod
        return elements

    @abstractmethod
    def forward(self, prediction, target):
        raise NotImplementedError


class VonMisesFisher2DLoss(VonMisesFisherLoss):
    """von Mises-Fisher loss function vectors in the 2D plane."""

    def forward(self, prediction, target):

        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 2
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]

        # Formatting target
        angle_true = target[:, 0]
        t = torch.stack(
            [
                torch.cos(angle_true),
                torch.sin(angle_true),
            ],
            dim=1,
        )

        # Formatting prediction
        angle_pred = prediction[:, 0]
        kappa = prediction[:, 1]
        p = kappa.unsqueeze(1) * torch.stack(
            [
                torch.cos(angle_pred),
                torch.sin(angle_pred),
            ],
            dim=1,
        )

        return self._evaluate(p, t)


class LogCoshLoss(_Loss):

    @classmethod
    def _log_cosh(cls, x):  # pylint: disable=invalid-name

        return x + softplus(-2.0 * x) - np.log(2.0)

    def forward(self, prediction, target):
        diff = prediction - target
        elements = self._log_cosh(diff)
        return elements
