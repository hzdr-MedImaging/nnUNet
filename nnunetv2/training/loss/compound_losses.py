import torch

from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss, edge_weight=None, edge_tf="lin", fg_weight=1):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        if edge_weight is not None:
            ce_kwargs['reduction'] = 'none'

            if edge_weight == 0:
                edge_tf = "lin"

            if edge_tf == "lin":
                self.edge_tfun = lambda x: x
            elif edge_tf == "sqr":
                self.edge_tfun = lambda x: x.square()
            elif edge_tf == "exp":
                self.edge_tfun = lambda x: torch.pow(2, x - 1)
            elif edge_tf == "sin":
                self.edge_tfun = lambda x: 1 + torch.sin((x - 1) / edge_weight * torch.pi/2)
            else:
                raise NotImplementedError

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.edge_size = edge_weight
        self.fg_weight = fg_weight


        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)


    def get_edge_weights(self, mask: torch.Tensor, edge_size: int):
        if edge_size is None or edge_size == 0:
            return torch.ones_like(mask)

        if mask.ndim == 4:
            pool_op = nn.MaxPool2d(3, stride=1, padding=1)
        elif mask.ndim == 5:
            pool_op = nn.MaxPool3d(3, stride=1, padding=1)
        else:
            raise NotImplementedError

        weights = torch.ones_like(mask) # start flat
        mask_expand = mask * 1.0
        mask_shrink = mask_expand
        # build mountains around edges iteratively
        for i in range(edge_size):
            mask_shrink = -pool_op(-mask_shrink)
            mask_expand = pool_op(mask_expand)
            weights = weights + mask_expand - mask_shrink # add evergrowing edge regions to the mask
        return weights


    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        if self.edge_size is None and self.fg_weight is None:
            dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
                if self.weight_dice != 0 else 0
            ce_loss = self.ce(net_output, target[:, 0]) \
                if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        else:
            # class 0 has to be background. Only edge between bg and fg is enhanced, not edges between classes
            fg_mask = target_dice > 0.5
            # fg_mask = torch.logical_or(fg_mask, net_output > 0.5)

            weights = torch.ones_like(target_dice)
            # weight edges
            if self.edge_size is not None:
                weights_eg = self.get_edge_weights(fg_mask, self.edge_size)
                weights_eg = self.edge_tfun(weights)
                weights = weights * weights_eg

            # weight the fg voxels
            if self.fg_weight is not None and self.fg_weight != 1:
                weights_fg = fg_mask * (self.fg_weight - 1) + 1
                weights = weights * weights_fg

            # add masking
            if mask is not None:
                weights = weights * mask

            weights = weights / torch.clip(weights.sum(), min=1e-8) * weights.numel()  # renormalize

            # this part should just work with weights instead of mask
            dc_loss = self.dc(net_output, target_dice, loss_mask=weights) \
                if self.weight_dice != 0 else 0
            # but here we need to do reduction ourselves
            ce_loss = (self.ce(net_output, target[:, 0]) * weights).sum() / torch.clip(weights.sum(), min=1e-8) \
                if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss, edge_weight=None, edge_tf="lin", fg_weight=None):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        if edge_weight is not None:
            raise NotImplementedError
        if fg_weight is not None:
            raise NotImplementedError

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_CE_weighted_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        print(dc_loss.shape, ce_loss.shape)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class CompositeLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super(CompositeLoss, self).__init__()
        assert isinstance(losses, list), "CompositeLoss class expects a list of losses for init."
        self.losses = losses
        if weights is None:
            weights = [1] * len(losses)
        assert len(losses) == len(weights), "CompositeLoss: len(losses) must equal len(weights)"
        self.weights = weights

    def forward(self, *arg_list):
        outputs = [loss(*args) for loss, args in zip(self.losses, arg_list)]
        return sum([output * weight for output, weight in zip(outputs, self.weights)])

if __name__ == '__main__':
    i1 = torch.randn(5, 3)
    i2 = torch.randn(5, 1)
    t1 = torch.randn(5)
    t2 = torch.randn(5, 1)

    i1 = torch.log(torch.softmax(i1, dim=1))
    t1 = torch.round(torch.sigmoid(t1 * 3) * 2).long()
    print(i1)
    print(t1)

    loss1 = nn.NLLLoss()
    loss2 = nn.BCEWithLogitsLoss()
    loss = CompositeLoss([loss1, loss2], weights=[1, 2])

    print(loss1(i1, t1), loss2(i2, t2))
    out = loss([i1, t1], [i2, t2])
    print(out)