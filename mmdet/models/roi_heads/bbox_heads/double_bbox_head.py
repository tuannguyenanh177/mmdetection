# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList

from mmdet.models.backbones.resnet import Bottleneck
from mmdet.models.builder import HEADS
from mmdet.models.losses import accuracy
from .bbox_head import BBoxHead


class BasicResBlock(BaseModule):
    """Basic residual block.

    This block is a little different from the block in the ResNet backbone.
    The kernel size of conv1 is 1 in this block while 3 in ResNet BasicBlock.

    Args:
        in_channels (int): Channels of the input feature map.
        out_channels (int): Channels of the output feature map.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(BasicResBlock, self).__init__(init_cfg)

        # main path
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        # identity path
        self.conv_identity = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)
        out = x + identity

        out = self.relu(out)
        return out


@HEADS.register_module()
class DoubleConvFCBBoxHead(BBoxHead):
    r"""Bbox head used in Double-Head R-CNN

    .. code-block:: none

                                          /-> cls
                      /-> shared convs ->
                                          \-> reg
        roi features
                                          /-> cls
                      \-> shared fc    ->
                                          \-> reg
    """  # noqa: W605

    def __init__(self,
                 num_convs=0,
                 num_fcs=0,
                 conv_out_channels=1024,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=dict(
                     type='Normal',
                     override=[
                         dict(type='Normal', name='fc_cls', std=0.01),
                         dict(type='Normal', name='fc_reg', std=0.001),
                         dict(
                             type='Xavier',
                             name='fc_branch',
                             distribution='uniform')
                     ]),
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(DoubleConvFCBBoxHead, self).__init__(init_cfg=init_cfg, **kwargs)
        assert self.with_avg_pool
        assert num_convs > 0
        assert num_fcs > 0
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # increase the channel of input features
        self.res_block = BasicResBlock(self.in_channels,
                                       self.conv_out_channels)

        # add conv heads
        self.conv_branch = self._add_conv_branch()
        # add fc heads
        self.fc_branch = self._add_fc_branch()

        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg_from_conv_head = nn.Linear(self.conv_out_channels, out_dim_reg)
        self.fc_reg_from_fc_head = nn.Linear(self.conv_out_channels, out_dim_reg)

        self.fc_cls_from_conv_head = nn.Linear(self.fc_out_channels, self.num_classes + 1)
        self.fc_cls_from_fc_head = nn.Linear(self.fc_out_channels, self.num_classes + 1)
        self.relu = nn.ReLU(inplace=True)

    def _add_conv_branch(self):
        """Add the fc branch which consists of a sequential of conv layers."""
        branch_convs = ModuleList()
        for i in range(self.num_convs):
            branch_convs.append(
                Bottleneck(
                    inplanes=self.conv_out_channels,
                    planes=self.conv_out_channels // 4,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        return branch_convs

    def _add_fc_branch(self):
        """Add the fc branch which consists of a sequential of fc layers."""
        branch_fcs = ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = (
                self.in_channels *
                self.roi_feat_area if i == 0 else self.fc_out_channels)
            branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        return branch_fcs

    def forward(self, x_cls, x_reg):
        # conv head
        x_conv = self.res_block(x_reg)

        for conv in self.conv_branch:
            x_conv = conv(x_conv)

        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)

        x_conv = x_conv.view(x_conv.size(0), -1)
        conv_bbox_pred = self.fc_reg_from_conv_head(x_conv)
        conv_cls_score = self.fc_cls_from_conv_head(x_conv)

        # fc head
        x_fc = x_cls.view(x_cls.size(0), -1)
        for fc in self.fc_branch:
            x_fc = self.relu(fc(x_fc))

        fc_cls_score = self.fc_cls_from_fc_head(x_fc)
        fc_bbox_pred = self.fc_reg_from_fc_head(x_fc)

        return fc_cls_score, fc_bbox_pred, conv_cls_score, conv_bbox_pred

    def loss(self,
             fc_cls_score,
             fc_bbox_pred,
             conv_cls_score,
             conv_bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):

        lamda_loss_fc = 0.7 # static number same as paper
        lamda_loss_conv = 0.8 # static number same as paper

        losses = dict()
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        if (fc_cls_score is not None) and (fc_bbox_pred is not None):
            if fc_cls_score.numel() > 0:
                loss_cls_from_fc = self.loss_cls(
                    fc_cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)

            loss_bbox_pred_from_fc = self.reuse_loss_bbox(
                fc_bbox_pred,
                rois,
                labels,
                bbox_targets,
                bbox_weights,
                reduction_override
            )

            loss_fc = lamda_loss_fc * loss_cls_from_fc + (1 - lamda_loss_fc) * loss_bbox_pred_from_fc
                
            if isinstance(loss_fc, dict):
                losses.update(loss_fc)
            else:
                losses['loss_of_fc'] = loss_fc
            if self.custom_activation:
                acc_ = self.loss_cls.get_accuracy(fc_cls_score, labels)
                losses.update(acc_)
            else:
                losses['acc'] = accuracy(fc_cls_score, labels)
        if (conv_cls_score is not None) and (conv_bbox_pred is not None):
            if conv_cls_score.numel() > 0:
                loss_cls_from_conv = self.loss_cls(
                    conv_cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                loss_bbox_pred_from_conv = self.reuse_loss_bbox(conv_bbox_pred, rois, labels, bbox_targets, bbox_weights, reduction_override)
            losses['loss_of_conv'] = (1 - lamda_loss_conv) * loss_cls_from_conv + lamda_loss_fc * loss_bbox_pred_from_conv
        return losses

    def reuse_loss_bbox(self, bbox_pred, rois, labels, bbox_targets, bbox_weights, reduction_override=None):
        bg_class_ind = self.num_classes
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        # do not perform bounding box regression for BG anymore.
        if pos_inds.any():
            if self.reg_decoded_bbox:
                # When the regression loss (e.g. `IouLoss`,
                # `GIouLoss`, `DIouLoss`) is applied directly on
                # the decoded bounding boxes, it decodes the
                # already encoded coordinates to absolute format.
                bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
            else:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), -1,
                    4)[pos_inds.type(torch.bool),
                    labels[pos_inds.type(torch.bool)]]
            return self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds.type(torch.bool)],
                bbox_weights[pos_inds.type(torch.bool)],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        else:
            return bbox_pred[pos_inds].sum()
