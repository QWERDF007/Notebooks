import math
from typing import Optional, Sequence
import warnings

import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, CheckpointLoader
from mmcv.utils.parrots_wrapper import _BatchNorm, SyncBatchNorm

from ...utils import get_root_logger
from ..builder import BACKBONES


class Block(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 stride=1, 
                 dilation=1, 
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=None,
                 norm_eval=False,
                 low_level_features=False,
                 start_with_relu=False,
                 no_skip=False):
        """ Xception Block

        Args:
            in_channels (int):
            out_channels (list or tuple):
            stride (int, optional): Conv stride. Defaults to 1.
            dilation (int, optional): Conv dilation rate. Defaults to 1.
            conv_cfg (dict, optional): Conv cfg. Defaults to None.
            norm_cfg (dict, optional): Batch norm cfg. Defaults to dict(type='BN', requires_grad=True).
            act_cfg (dict, optional): Activation functions cfg. Defaults to None.
            norm_eval (bool, optional): unknown. Defaults to False.
            low_level_features (bool, optional): whether output low level features or not. Defaults to False.
            start_with_relu (bool, optional): whether start with relu or not. Defaults to False.
            no_skip (bool, optional): no shortcut connection. Defaults to False.
        """
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels[-1]
        self.no_skip = no_skip
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.num_units = len(out_channels)
        self.low_level_features = low_level_features
        
        if not no_skip and (self.out_channels != self.in_channels or stride != 1):
            self.skip = build_conv_layer(
                    self.conv_cfg,
                    self.in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False)
            
            self.skipbn = build_norm_layer(self.norm_cfg, self.out_channels)[1]
        else:
            self.skip = None
            self.skipbn = None

        
        units = []
        if start_with_relu:
            units.append(nn.ReLU(inplace=True))
        for i in range(self.num_units):
            inc = in_channels if i == 0 else out_channels[i-1]
            outc = out_channels[i]
            units.append(
                DepthwiseSeparableConvModule(
                    inc, 
                    outc, 
                    3, 
                    stride=1 if i < (self.num_units - 1) else stride, 
                    dilation=dilation, 
                    padding=dilation, 
                    norm_cfg=self.norm_cfg,
                    act_cfg=act_cfg))
            
            if act_cfg is None and i < self.num_units - 1:
                units.append(nn.ReLU(inplace=True))

        self.units = nn.Sequential(*units)

    def forward(self, x):
        skip = x
        low_level_features = None

        for i, unit in enumerate(self.units):
            x = unit(x)
            if self.low_level_features and i == self.num_units - 2:
                low_level_features = x

        if self.skip is not None:
            skip = self.skip(skip)
            skip = self.skipbn(skip)
        
        if not self.no_skip:
            x = x + skip

        return x, low_level_features


def xception_arch(name, strides=(2,2,2,1), dilations=(1,1,1,2)):
    arch  = {
        'xception65': [
            # entry flow
            dict(in_channels=64, out_channels=(128, 128, 128), stride=strides[0], dilation=1, start_with_relu=False),
            dict(in_channels=128, out_channels=(256, 256, 256), stride=strides[1], dilation=1, start_with_relu=True, low_level_features=True),
            dict(in_channels=256, out_channels=(728, 728, 728), stride=strides[2], dilation=dilations[0], start_with_relu=True),
            # middle flow
            *([dict(in_channels=728, out_channels=(728, 728, 728), stride=1, dilation=dilations[1], start_with_relu=True)] * 16),
            # exit flow
            dict(in_channels=728, out_channels=(728, 1024, 1024), stride=strides[3], dilation=dilations[2], start_with_relu=True),
            dict(in_channels=1024, out_channels=(1536, 1536, 2048), stride=1, dilation=dilations[3], act_cfg=dict(type='ReLU'), start_with_relu=False, no_skip=True),
        ]
    }
    assert name.lower() in arch.keys(), "Not implement arch: {}".format(name)
    return arch[name]


class AlignedXception(BaseModule):
    """AlignedXception backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Number of stem channels. Default: 64.
        base_channels (int): Number of base channels of res layer. Default: 64.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: (2, 2, 2, 1).
        dilations (Sequence[int]): Dilation of each stage.
            Default: (1, 1, 1, 2).
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): Dictionary to construct and config conv layer.
            When conv_cfg is None, cfg will be set to dict(type='Conv2d').
            Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        multi_grid (Sequence[int]|None): Multi grid dilation rates of last
            stage. Default: None.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                #  depth,
                 in_channels=3,
                 stem_channels=64,
                #  base_channels=64,
                 strides=(2, 2, 2, 1),
                 dilations=(1, 1, 1, 2),
                 out_indices=(0, 1, 18, 20),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 dcn=None,
                 multi_grid=None,
                 pretrained=None,
                 init_cfg=None):

        super(AlignedXception, self).__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'

        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.stem_channels = stem_channels
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.multi_grid = multi_grid
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        blocks_cfg = xception_arch('xception65', dilations=dilations)
        self.blocks = []
        for i, b in enumerate(blocks_cfg):
            block_name = f'block{i+1}'
            block = Block(conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, norm_eval=self.norm_eval, **b)
            self.add_module(block_name, block)
            self.blocks.append(block_name)
    
    def _make_stem_layer(self, in_channels, stem_channels):
        """Make stem layer for AlignedXception."""
        self.stem = nn.Sequential()
        self.stem.add_module(
            'conv1',
            build_conv_layer(self.conv_cfg,
                in_channels,
                stem_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False)
        )
        self.stem.add_module(
            'bn1',
            build_norm_layer(self.norm_cfg, stem_channels // 2)[1]
        )
        self.stem.add_module('relu1', nn.ReLU(inplace=True))
        self.stem.add_module(
            'conv2',
            build_conv_layer(self.conv_cfg,
                stem_channels // 2,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False)
        )
        self.stem.add_module(
            'bn2',
            build_norm_layer(self.norm_cfg, stem_channels)[1]
        )
        self.stem.add_module('relu2', nn.ReLU(inplace=True))

    

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, block_name in enumerate(self.blocks):
            block = getattr(self, block_name)
            x, low_level_features = block(x)
            if i in self.out_indices:
                if len(outs) == 0:
                    outs.append(low_level_features)
                else:
                    outs.append(x)
        return tuple(outs)
    
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(AlignedXception, self).train(mode)
        # self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def init_weights(self):
        logger = get_root_logger()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        if self.init_cfg is not None:
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
            model_dict = self.state_dict()
            converted_state_dict = dict() 
            for k, v in ckpt.items():
                if ('stem.' + k) in model_dict:
                    converted_state_dict['stem.' + k] = v
                elif k.startswith('block'):
                    if k.startswith('block11'):
                        if 'block11.rep.1.conv1.weight' ==  k:
                            converted_state_dict['block12.units.1.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block13.units.1.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block14.units.1.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block15.units.1.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block16.units.1.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block17.units.1.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block18.units.1.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block19.units.1.depthwise_conv.conv.weight'] = v
                        elif 'block11.rep.1.pointwise.weight' == k:
                            converted_state_dict['block12.units.1.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block13.units.1.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block14.units.1.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block15.units.1.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block16.units.1.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block17.units.1.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block18.units.1.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block19.units.1.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                        elif 'block11.rep.4.conv1.weight' == k:
                            converted_state_dict['block12.units.3.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block13.units.3.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block14.units.3.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block15.units.3.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block16.units.3.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block17.units.3.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block18.units.3.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block19.units.3.depthwise_conv.conv.weight'] = v
                        elif 'block11.rep.4.pointwise.weight' == k:
                            converted_state_dict['block12.units.3.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block13.units.3.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block14.units.3.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block15.units.3.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block16.units.3.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block17.units.3.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block18.units.3.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block19.units.3.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                        elif 'block11.rep.7.conv1.weight' == k:
                            converted_state_dict['block12.units.5.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block13.units.5.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block14.units.5.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block15.units.5.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block16.units.5.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block17.units.5.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block18.units.5.depthwise_conv.conv.weight'] = v
                            converted_state_dict['block19.units.5.depthwise_conv.conv.weight'] = v
                        elif 'block11.rep.7.pointwise.weight' == k:
                            converted_state_dict['block12.units.5.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block13.units.5.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block14.units.5.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block15.units.5.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block16.units.5.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block17.units.5.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block18.units.5.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                            converted_state_dict['block19.units.5.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                    elif k.startswith('block12'):
                        if 'skip' in k:
                            converted_state_dict[k.replace('block12', 'block20')] = v
                        elif 'block12.rep.1.conv1.weight' == k:
                            converted_state_dict['block20.units.1.depthwise_conv.conv.weight'] = v
                        elif 'block12.rep.1.pointwise.weight' == k:
                            converted_state_dict['block20.units.1.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                        elif 'block12.rep.4.conv1.weight' == k:
                            converted_state_dict['block20.units.3.depthwise_conv.conv.weight'] = v
                        elif 'block12.rep.4.pointwise.weight' == k:
                            converted_state_dict['block20.units.3.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
                    else:
                        block_name = k.split('.')[0]
                        if 'skip' in k:
                            converted_state_dict[k] = v
                        elif block_name == 'block1':
                            if 'rep.0.conv1.weight' in k:
                                converted_state_dict['{}.units.0.depthwise_conv.conv.weight'.format(block_name)] = v
                            elif 'rep.0.pointwise.weight' in k:
                                converted_state_dict['{}.units.0.pointwise_conv.conv.weight'.format(block_name)] = v.unsqueeze(-1).unsqueeze(-1)
                            elif 'rep.3.conv1.weight' in k:
                                converted_state_dict['{}.units.2.depthwise_conv.conv.weight'.format(block_name)] = v
                            elif 'rep.3.pointwise.weight' in k:
                                converted_state_dict['{}.units.2.pointwise_conv.conv.weight'.format(block_name)] = v.unsqueeze(-1).unsqueeze(-1)
                        elif block_name in ('block2', 'block3'):
                            if 'rep.1.conv1.weight' in k:
                                converted_state_dict['{}.units.1.depthwise_conv.conv.weight'.format(block_name)] = v
                            elif 'rep.1.pointwise.weight' in k:
                                converted_state_dict['{}.units.1.pointwise_conv.conv.weight'.format(block_name)] = v.unsqueeze(-1).unsqueeze(-1)
                            elif 'rep.4.conv1.weight' in k:
                                converted_state_dict['{}.units.3.depthwise_conv.conv.weight'.format(block_name)] = v
                            elif 'rep.4.pointwise.weight' in k:
                                converted_state_dict['{}.units.3.pointwise_conv.conv.weight'.format(block_name)] = v.unsqueeze(-1).unsqueeze(-1)
                        else:
                            if 'rep.1.conv1.weight' in k:
                                converted_state_dict['{}.units.1.depthwise_conv.conv.weight'.format(block_name)] = v
                            elif 'rep.1.pointwise.weight' in k:
                                converted_state_dict['{}.units.1.pointwise_conv.conv.weight'.format(block_name)] = v.unsqueeze(-1).unsqueeze(-1)
                            elif 'rep.4.conv1.weight' in k:
                                converted_state_dict['{}.units.3.depthwise_conv.conv.weight'.format(block_name)] = v
                            elif 'rep.4.pointwise.weight' in k:
                                converted_state_dict['{}.units.3.pointwise_conv.conv.weight'.format(block_name)] = v.unsqueeze(-1).unsqueeze(-1)
                            elif 'rep.7.conv1.weight' in k:
                                converted_state_dict['{}.units.5.depthwise_conv.conv.weight'.format(block_name)] = v
                            elif 'rep.7.pointwise.weight' in k:
                                converted_state_dict['{}.units.5.pointwise_conv.conv.weight'.format(block_name)] = v.unsqueeze(-1).unsqueeze(-1)
                elif k.startswith('conv4'):
                    if 'conv1' in k:
                        converted_state_dict['block21.units.2.depthwise_conv.conv.weight'] = v
                    elif 'pointwise' in k:
                        converted_state_dict['block21.units.2.pointwise_conv.conv.weight'] = v.unsqueeze(-1).unsqueeze(-1)
            model_dict.update(converted_state_dict)
            self.load_state_dict(model_dict)


@BACKBONES.register_module()
class Xception(AlignedXception):
    def __init__(self, **kwargs) -> AlignedXception:
        super(Xception, self).__init__(**kwargs)

    