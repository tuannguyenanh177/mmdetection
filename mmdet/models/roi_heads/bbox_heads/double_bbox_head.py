# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.runner import BaseModule, ModuleList

from mmdet.models.backbones.resnet import Bottleneck
from mmdet.models.builder import HEADS
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
        self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg)

        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes + 1)
        self.relu = nn.ReLU(inplace=True)

        _, self.norm1 = build_norm_layer(dict(type='BN1d'), 1024)
        _, self.norm2 = build_norm_layer(dict(type='BN'), 1024)
        self.upsample = nn.Upsample(scale_factor=7, mode='nearest')

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
        # print(f'size of x_cls {x_cls.size()}, x_reg {x_reg.size()}')
    
        fc_1 = self.fc_branch[0]
        fc_2 = self.fc_branch[1]

        conv_first_2_branchs = self.conv_branch[:2]
        conv_last_2_branchs = self.conv_branch[2:]

        for conv in conv_first_2_branchs:
            x_conv = conv(x_conv)

        # print(f'size of x_conv {x_conv.size()}')

        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)

        x_conv_to_fc = x_conv.view(x_conv.size(0), -1)
        
        x_fc = x_cls.view(x_cls.size(0), -1)
        x_fc = self.relu(fc_1(x_fc))

        x_fc_to_conv = x_fc.view(x_fc.size(0), x_fc.size(1), 1, 1)
        x_fc_to_conv = self.upsample(x_fc_to_conv)

        # put conv information to fc_2
        x_fc = x_fc + x_conv_to_fc
        x_fc = self.relu(self.norm1(x_fc))
        x_fc = self.relu(fc_2(x_fc))

        # put fc information to last 2 conv branchs
        x_conv = x_conv + x_fc_to_conv
        x_conv = self.relu(self.norm2(x_conv))
        for conv in conv_last_2_branchs:
            x_conv = conv(x_conv)
        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)

        x_conv = x_conv.view(x_conv.size(0), -1)

        bbox_pred = self.fc_reg(x_conv)
        cls_score = self.fc_cls(x_fc)

        return cls_score, bbox_pred
