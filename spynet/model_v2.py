import requests
from pathlib import Path
from typing import Sequence, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

import spynet
from mmcv.cnn import ConvModule
from mmengine import MMLogger, print_log
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint

from .flow_warp import flow_warp

class SpyNetUnit(nn.Module):

    def __init__(self, input_channels: int = 8):
        super(SpyNetUnit, self).__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))
        
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.weight = m.weight / 100
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, 
                frames: Tuple[torch.Tensor, torch.Tensor], 
                optical_flow: torch.Tensor = None,
                upsample_optical_flow: bool = True) -> torch.Tensor:
        f_frame, s_frame = frames

        if optical_flow is None:
            # If optical flow is None (k = 0) then create empty one having the
            # same size as the input frames, therefore there is no need to 
            # upsample it later
            upsample_optical_flow = False
            b, c, h, w = f_frame.size()
            optical_flow = torch.zeros(b, 2, h, w, device=s_frame.device)

        if upsample_optical_flow:
            optical_flow = F.interpolate(
                optical_flow, scale_factor=2, align_corners=True, 
                mode='bilinear') * 2

        s_frame = spynet.nn.warp(s_frame, optical_flow, s_frame.device)
        s_frame = torch.cat([s_frame, optical_flow], dim=1)
        
        inp = torch.cat([f_frame, s_frame], dim=1)
        return self.basic_module(inp)


class SpyNet(nn.Module):

    def __init__(self, units: Sequence[SpyNetUnit] = None, k: int = None):
        super(SpyNet, self).__init__()
        
        if units is not None and k is not None:
            assert len(units) == k

        if units is None and k is None:
            raise ValueError('At least one argument (units or k) must be' 
                             'specified')
        # self.register_buffer(
        #     'mean',
        #     torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer(
        #     'std',
        #     torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if units is not None:
            self.basic_module = nn.ModuleList(units)
        else:
            units = [SpyNetUnit() for _ in range(k)]
            self.basic_module = nn.ModuleList(units)

    def forward(self, 
                frames: Tuple[torch.Tensor, torch.Tensor],
                limit_k: int = -1) -> torch.Tensor:
        """
        Parameters
        ----------
        frames: Tuple[torch.Tensor, torch.Tensor]
            Highest resolution frames. Each tuple element has shape
            [BATCH, 3, HEIGHT, WIDTH]
        """
        if limit_k == -1:
            units = self.basic_module
        else:
            units = self.basic_module[:limit_k]
        Vk_1 = None

        for k, G in enumerate(self.basic_module):
            im_size = spynet.config.GConf(k).image_size
            x1 = F.interpolate(frames[0], im_size, mode='bilinear',
                               align_corners=True)
            x2 = F.interpolate(frames[1], im_size, mode='bilinear',
                               align_corners=True)

            if Vk_1 is not None: # Upsample the previous optical flow
                Vk_1 = F.interpolate(
                    Vk_1, scale_factor=2, align_corners=True, 
                    mode='bilinear') * 2.

            Vk = G((x1, x2), Vk_1, upsample_optical_flow=False)
            Vk_1 = Vk + Vk_1 if Vk_1 is not None else Vk
        
        return Vk_1

    @classmethod
    def from_pretrained(cls: Type['SpyNet'], 
                        name: str, 
                        map_location: torch.device = torch.device('cpu'),
                        dst_file: str = None) -> 'SpyNet':
        
        def get_model(path: str) -> 'SpyNet':
            checkpoint = torch.load(path, 
                                    map_location=map_location)
            k = 6

            instance = cls(k=k)
            instance.load_state_dict(checkpoint, strict=True)
            instance.to(map_location)
            return instance

        return get_model(str(name))