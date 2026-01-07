import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PatchCoreBackbone(nn.Module):
    """
    与训练阶段一致的 PatchCore backbone：
    - ResNet / Wide-ResNet
    - 提取 layer1/2/3/4 的特征并在通道维拼接
    """

    def __init__(
        self,
        backbone_name: str,
        layers: Tuple[int, ...],
        pretrained: bool = True,
        custom_weight_path: Optional[str] = None,
    ):
        super().__init__()
        self.layers = layers

        # 选择 backbone
        if backbone_name == "wide_resnet50_2":
            self.backbone = models.wide_resnet50_2(pretrained=pretrained)
        elif backbone_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone_name == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # 可选：加载自定义权重（你现有逻辑保持不变）
        if custom_weight_path is not None and os.path.isfile(custom_weight_path):
            print(f"[INFO] Loading custom weight: {custom_weight_path}")
            try:
                state = torch.load(custom_weight_path, map_location="cpu", weights_only=False)
            except TypeError:
                state = torch.load(custom_weight_path, map_location="cpu")

            if isinstance(state, torch.nn.Module):
                print("[INFO] Loaded a model object, using its state_dict")
                state = state.state_dict()

            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            if isinstance(state, dict):
                from collections import OrderedDict
                new_state = OrderedDict()
                for k, v in state.items():
                    new_state[k.replace("module.", "")] = v
                state = new_state

            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            if missing:
                print(f"[WARN] Missing keys when loading weights: {missing}")
            if unexpected:
                print(f"[WARN] Unexpected keys when loading weights: {unexpected}")
        else:
            print("[INFO] No custom weights provided. Using torchvision pretrained weights.")

        # 冻结参数（PatchCore 常规做法）
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet 标准流程
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        out1 = self.backbone.layer1(x)
        out2 = self.backbone.layer2(out1)
        out3 = self.backbone.layer3(out2)
        out4 = self.backbone.layer4(out3)

        feats = []
        if 1 in self.layers: feats.append(out1)
        if 2 in self.layers: feats.append(out2)
        if 3 in self.layers: feats.append(out3)
        if 4 in self.layers: feats.append(out4)

        if not feats:
            raise RuntimeError("No layers selected in PatchCoreBackbone.")

        # 对齐到同一分辨率后拼接
        ref_h, ref_w = feats[0].shape[2], feats[0].shape[3]
        upsampled = []
        for f in feats:
            if f.shape[2] != ref_h or f.shape[3] != ref_w:
                f = F.interpolate(f, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
            upsampled.append(f)

        return torch.cat(upsampled, dim=1)
