# utils.py
#
# source code for utility functions

# imports
import torch
from torch.distributions import kl_divergence, Normal
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import functional as F

# class and function definitions and implementations
class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer,
                                               gamma,
                                               last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]


def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))

    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()

    return N


def loss_function(recon, recon_rhythm, target_tensor,
                  rhythm_target, distribution_1,
                  distribution_2, step, beta=.1):

    CE1 = F.nll_loss(
        recon.view(-1, recon.size(-1)),
        target_tensor,
        reduction="elementwise_mean"
    )
    CE2 = F.nll_loss(
        recon_rhythm.view(-1, recon_rhythm.size(-1)),
        rhythm_target,
        reduction="elementwise_mean"
    )

    normal1 = std_normal(distribution_1.mean.size())
    normal2 = std_normal(distribution_2.mean.size())

    KLD1 = kl_divergence(distribution_1, normal1).mean()
    KLD2 = kl_divergence(distribution_2, normal2).mean()

    return CE1 + CE2 + beta * (KLD1 + KLD2)
