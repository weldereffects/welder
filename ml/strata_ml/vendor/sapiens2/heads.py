# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Welder: the network parts of Sapiens2's NormalHead and SegHead (sapiens/dense/src/models/heads),
# without the training losses or the model registry, plus the two estimators the checkpoints were
# saved from (state dict keys "backbone.*" and "decode_head.*").

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .backbone import Sapiens2

EMBED_DIMS = {"sapiens2_0.4b": 1024, "sapiens2_0.8b": 1280, "sapiens2_1b": 1536, "sapiens2_5b": 2432}
IMAGE_SIZE = (1024, 768)  # height x width, the size every Sapiens2 dense model was trained at
MEAN = (123.675, 116.28, 103.53)  # RGB, 0-255
STD = (58.395, 57.12, 57.375)


class NormalHead(nn.Module):
    def __init__(self, in_channels: int = 768, upsample_channels: List[int] = (768, 384, 192, 96),
                 conv_out_channels: Optional[Sequence[int]] = None, conv_kernel_sizes: Optional[Sequence[int]] = None):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        up_blocks = []
        cur = in_channels
        for out_ch in upsample_channels:
            up_blocks.append(nn.Sequential(
                nn.Conv2d(cur, out_ch * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.InstanceNorm2d(out_ch),
                nn.SiLU(inplace=True),
            ))
            cur = out_ch
        self.upsample_blocks = nn.Sequential(*up_blocks)
        conv_layers = []
        if conv_out_channels and conv_kernel_sizes:
            for out_ch, k in zip(conv_out_channels, conv_kernel_sizes):
                conv_layers += [nn.Conv2d(cur, out_ch, k, padding=(k - 1) // 2), nn.InstanceNorm2d(out_ch), nn.SiLU(inplace=True)]
                cur = out_ch
        self.conv_layers = nn.Sequential(*conv_layers)
        self.conv_normal = nn.Conv2d(cur, 3, kernel_size=1)

    def forward(self, x):
        return self.conv_normal(self.conv_layers(self.upsample_blocks(self.input_conv(x))))


class SegHead(nn.Module):
    def __init__(self, in_channels: int = 768, deconv_out_channels=(512, 256, 128, 64), deconv_kernel_sizes=(4, 4, 4, 4),
                 conv_out_channels=(64, 64), conv_kernel_sizes=(1, 1), num_classes: int = 29):
        super().__init__()
        layers = []
        cur = in_channels
        for out_ch, k in zip(deconv_out_channels, deconv_kernel_sizes):
            padding, output_padding = {4: (1, 0), 3: (1, 1), 2: (0, 0)}[k]
            layers += [nn.ConvTranspose2d(cur, out_ch, kernel_size=k, stride=2, padding=padding,
                                          output_padding=output_padding, bias=False),
                       nn.InstanceNorm2d(out_ch), nn.SiLU(inplace=True)]
            cur = out_ch
        self.deconv_layers = nn.Sequential(*layers)
        layers = []
        for out_ch, k in zip(conv_out_channels, conv_kernel_sizes):
            layers += [nn.Conv2d(cur, out_ch, kernel_size=k, stride=1, padding=(k - 1) // 2),
                       nn.InstanceNorm2d(out_ch), nn.SiLU(inplace=True)]
            cur = out_ch
        self.conv_layers = nn.Sequential(*layers)
        self.conv_seg = nn.Conv2d(cur, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv_seg(self.conv_layers(self.deconv_layers(x)))


def _backbone(arch):
    return Sapiens2(arch=arch, img_size=IMAGE_SIZE, patch_size=16, final_norm=True, use_tokenizer=False,
                    with_cls_token=True, out_type="featmap")


class NormalEstimator(nn.Module):
    """Image (B x 3 x 1024 x 768, normalised) -> normals (B x 3 x 1024 x 768, not yet unit length)."""

    def __init__(self, arch="sapiens2_1b"):
        super().__init__()
        self.backbone = _backbone(arch)
        up = [1536, 768, 512, 256] if arch == "sapiens2_5b" else [768, 512, 256, 128]
        self.decode_head = NormalHead(EMBED_DIMS[arch], up, (64, 32, 16), (3, 3, 3))

    def forward(self, x):
        return self.decode_head(self.backbone(x)[0])


class SegEstimator(nn.Module):
    """Image (B x 3 x 1024 x 768, normalised) -> logits for the 29 classes (B x 29 x 1024 x 768)."""

    def __init__(self, arch="sapiens2_1b"):
        super().__init__()
        self.backbone = _backbone(arch)
        self.decode_head = SegHead(EMBED_DIMS[arch])

    def forward(self, x):
        return self.decode_head(self.backbone(x)[0])


def load(kind, arch, path, device="cpu", dtype=torch.float32):
    """A Sapiens2 dense model ('normal' or 'seg') with its weights (.safetensors or .pth)."""
    net = NormalEstimator(arch) if kind == "normal" else SegEstimator(arch)
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(str(path), device="cpu")
    else:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith(("decode_head.loss", "data_preprocessor"))}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing:
        raise RuntimeError(f"the Sapiens2 {kind} checkpoint doesn't fit the {arch} network: missing "
                           f"{len(missing)} weights (e.g. {missing[0]}). Is it the right model size?")
    return net.to(device=device, dtype=dtype).eval()
