from typing import Dict
import torch, warnings
import torch.nn as nn
import torch.nn.functional as F


class AUCMLoss(torch.nn.Module):
    r"""
        AUC-Margin loss with squared-hinge surrogate loss for optimizing AUROC. The objective function is defined as:

        .. math::

            \min _{\substack{\mathbf{w} \in \mathbb{R}^d \\(a, b) \in \mathbb{R}^2}} \max _{\alpha \in \mathbb{R^+}} f(\mathbf{w}, a, b, \alpha):=\mathbb{E}_{\mathbf{z}}[F(\mathbf{w}, a, b, \alpha ; \mathbf{z})]

        where

        .. math::

            F(\mathbf{w},a,b,\alpha; \mathbf{z}) &=(1-p)(h_{\mathbf{w}}(x)-a)^2\mathbb{I}_{[y=1]} +p(h_{\mathbf{w}}(x)-b)^2\mathbb{I}_{[y=-1]} \\
            &+2\alpha(p(1-p)m+ p h_{\mathbf{w}}(x)\mathbb{I}_{[y=-1]}-(1-p)h_{\mathbf{w}}(x)\mathbb{I}_{[y=1]})\\
            &-p(1-p)\alpha^2

        :math:`h_{\mathbf{w}}` is the prediction scoring function, e.g., deep neural network, :math:`p` is the ratio of positive samples to all samples, :math:`a`, :math:`b` are the running statistics of
        the positive and negative predictions, :math:`\alpha` is the auxiliary variable derived from the problem formulation and :math:`m` is the margin term. We denote this version of AUCMLoss as ``v1``.

        To remove the class prior :math:`p` in the above formulation, we can write the new objective function as follow:

         .. math::

            f(\mathbf{w},a,b,\alpha) &= \mathbb{E}_{y=1}[(h_{\mathbf{w}}(x)-a)^2] + \mathbb{E}_{y=-1}[(h_{\mathbf{w}}(x)-b)^2] \\
            &+2\alpha(m + \mathbb{E}_{y=-1}[h_{\mathbf{w}}(x)] - \mathbb{E}_{y=1}[h_{\mathbf{w}}(x)])\\
            &-\alpha^2

        We denote this version of AUCMLoss as ``v2``. The optimization algorithm for solving the above objectives are implemented as :obj:`~libauc.optimizers.PESG`. For the derivations, please refer to the original paper [1]_.

        args:
            margin (float): margin for squared-hinge surrogate loss (default: ``1.0``).
            imratio (float, optional): the ratio of the number of positive samples to the number of total samples in the training dataset.
                                       If this value is not given, the mini-batch statistics will be used instead.
            version (str, optional): whether to include prior :math:`p` in the objective function (default: ``'v1'``).


        Example:
            >>> loss_fn = libauc.losses.AUCMLoss(margin=1.0)
            >>> preds = torch.randn(32, 1, requires_grad=True)
            >>> target = torch.empty(32, dtype=torch.long).random_(1)
            >>> loss = loss_fn(preds, target)
            >>> loss.backward()

        .. note::
            To use ``v2`` of AUCMLoss, plesae set ``version='v2'``. Otherwise, the default version is ``v1``. The ``v2`` version requires the use of :obj:`~libauc.sampler.DualSampler`.

        .. note::
            Practial Tips:

            - ``epoch_decay`` is a regularization parameter similar to `weight_decay` that can be tuned in the same range.
            - For complex tasks, it is recommended to use regular loss to pretrain the model, and then switch to AUCMLoss for finetuning with a smaller learning rate.

        Reference:
            .. [1] Yuan, Zhuoning, Yan, Yan, Sonka, Milan, and Yang, Tianbao.
               "Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification."
               Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
               https://arxiv.org/abs/2012.03173
    """

    def __init__(self, margin=1.0, imratio=None, version="v1", device=None):
        super(AUCMLoss, self).__init__()
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.margin = margin
        self.p = imratio
        self.version = version
        assert version in [
            "v1",
            "v2",
        ], "Input value is not valid! Possible values are ['v1', 'v2']."
        self.a = torch.zeros(
            1, dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.b = torch.zeros(
            1, dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha = torch.zeros(
            1, dtype=torch.float32, device=self.device, requires_grad=True
        )

    def mean(self, tensor):
        return torch.sum(tensor) / torch.count_nonzero(tensor)

    def forward(self, y_pred, y_true, auto=True, **kwargs):
        pos_mask = (1 == y_true).float()
        neg_mask = (0 == y_true).float()

        if sum(pos_mask) == 0:
            warnings.warn(
                "Input data has no positive sample! Please use 'libauc.sampler.DualSampler' for data resampling!",
                UserWarning,
            )

        if self.version == "v1":
            if auto or self.p == None:
                self.p = pos_mask.sum() / y_true.shape[0]
            loss = (
                (1 - self.p)
                * torch.mean((y_pred - self.a) ** 2 * (1 == y_true).float())
                + self.p * torch.mean((y_pred - self.b) ** 2 * (0 == y_true).float())
                + 2
                * self.alpha
                * (
                    self.p * (1 - self.p) * self.margin
                    + torch.mean(
                        (
                            self.p * y_pred * (0 == y_true).float()
                            - (1 - self.p) * y_pred * (1 == y_true).float()
                        )
                    )
                )
                - self.p * (1 - self.p) * self.alpha**2
            )
        else:
            loss = (
                self.mean((y_pred - self.a) ** 2 * pos_mask)
                + self.mean((y_pred - self.b) ** 2 * neg_mask)
                + 2
                * self.alpha
                * (
                    self.margin
                    + self.mean((y_pred * neg_mask) - self.mean(y_pred * pos_mask))
                )
                - self.alpha**2
            )
        return loss


class WeightedSumLosses(nn.Module):
    def __init__(
        self, loss_dict: nn.ModuleDict, weights_dict: Dict[str, float], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_dict = loss_dict
        self.weighted_dict = weights_dict

    def forward(self, pred, tgt):
        # total_loss = torch.tensor(0.0, dtype=pred.dtype, requires_grad=True)
        # for k in self.loss_dict.keys():
        #     l = self.loss_dict[k](F.sigmoid(pred), tgt)
        #     weighted_loss = self.weighted_dict[k] * l
        #     total_loss += weighted_loss

        # return weighted_loss
        return sum(
            [
                self.loss_dict[k](F.sigmoid(pred), tgt) * self.weighted_dict[k]
                for k in self.loss_dict.keys()
            ]
        )


class SigmoidWrapper(nn.Module):
    def __init__(self, loss_fn: nn.Module) -> None:
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, tgt):
        return self.loss_fn(F.sigmoid(pred), tgt)
