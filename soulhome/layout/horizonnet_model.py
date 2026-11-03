"""HorizonNet network (ResNet-50 + Bi-LSTM), vendored and trimmed from
https://github.com/sunset1995/HorizonNet (MIT License, Copyright (c) 2019 Cheng Sun).

Changes from upstream: ResNet backbone only, no ImageNet weight download at construction (the
checkpoint overwrites every parameter), no distutils. Module structure is unchanged so the official
checkpoints (``sunset1995/HorizonNet`` on the Hugging Face Hub) load with strict=True.
"""

from __future__ import annotations

import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

HF_REPO = "sunset1995/HorizonNet"
# Checkpoints that load with torch's safe (weights_only) unpickler. The panos2d3d checkpoint embeds
# numpy objects and would need a full unpickle, so it is deliberately not offered.
CHECKPOINTS = {
    "mp3d": "resnet50_rnn__mp3d.pth",  # Matterport3D: real furnished homes, non-cuboid (default)
    "zind": "resnet50_rnn__zind.pth",  # Zillow Indoor
    "st3d": "resnet50_rnn__st3d.pth",  # Structured3D (synthetic)
}


class LRPad(nn.Module):
    """Circular padding along width: the panorama wraps around."""

    def __init__(self, padding=1):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        p = self.padding
        return torch.cat([x[..., -p:], x, x[..., :p]], dim=3)


def wrap_lr_pad(net):
    for name, m in list(net.named_modules()):
        if not isinstance(m, nn.Conv2d) or m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = (m.padding[0], 0)
        names = name.split(".")
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(root, names[-1], nn.Sequential(LRPad(w_pad), m))


class Resnet(nn.Module):
    def __init__(self, backbone="resnet50"):
        super().__init__()
        self.encoder = getattr(models, backbone)(weights=None)
        del self.encoder.fc, self.encoder.avgpool

    def forward(self, x):
        e = self.encoder
        x = e.maxpool(e.relu(e.bn1(e.conv1(x))))
        feats = []
        for layer in (e.layer1, e.layer2, e.layer3, e.layer4):
            x = layer(x)
            feats.append(x)
        return feats


class ConvCompressH(nn.Module):
    def __init__(self, in_c, out_c, ks=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks // 2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.layer = nn.Sequential(
            ConvCompressH(in_c, in_c // 2),
            ConvCompressH(in_c // 2, in_c // 2),
            ConvCompressH(in_c // 2, in_c // 4),
            ConvCompressH(in_c // 4, out_c),
        )

    def forward(self, x, out_w):
        x = self.layer(x)
        factor = out_w // x.shape[3]
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
        x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode="bilinear", align_corners=False)
        return x[..., factor:-factor]


class GlobalHeightStage(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_scale=8):
        super().__init__()
        self.cs = c1, c2, c3, c4
        self.ghc_lst = nn.ModuleList([GlobalHeightConv(c, c // out_scale) for c in (c1, c2, c3, c4)])

    def forward(self, conv_list, out_w):
        bs = conv_list[0].shape[0]
        return torch.cat([f(x, out_w).reshape(bs, -1, out_w) for f, x in zip(self.ghc_lst, conv_list)], dim=1)


class HorizonNet(nn.Module):
    def __init__(self, backbone="resnet50", use_rnn=True):
        super().__init__()
        self.backbone, self.use_rnn = backbone, use_rnn
        self.out_scale, self.step_cols, self.rnn_hidden_size = 8, 4, 512
        self.register_buffer("x_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("x_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.feature_extractor = Resnet(backbone)
        with torch.no_grad():
            c1, c2, c3, c4 = [b.shape[1] for b in self.feature_extractor(torch.zeros(1, 3, 512, 1024))]
        c_last = (c1 * 8 + c2 * 4 + c3 * 2 + c4 * 1) // self.out_scale
        self.reduce_height_module = GlobalHeightStage(c1, c2, c3, c4, self.out_scale)
        self.bi_rnn = nn.LSTM(input_size=c_last, hidden_size=self.rnn_hidden_size, num_layers=2, dropout=0.5,
                              batch_first=False, bidirectional=True)
        self.drop_out = nn.Dropout(0.5)
        self.linear = nn.Linear(2 * self.rnn_hidden_size, 3 * self.step_cols)
        wrap_lr_pad(self)

    def forward(self, x):
        """x: (B, 3, 512, 1024) RGB in [0, 1] -> (y_bon (B, 2, 1024), y_cor (B, 1, 1024))."""
        x = (x[:, :3] - self.x_mean) / self.x_std
        feature = self.reduce_height_module(self.feature_extractor(x), x.shape[3] // self.step_cols)
        feature = feature.permute(2, 0, 1)
        output, _ = self.bi_rnn(feature)
        output = self.linear(self.drop_out(output))
        output = output.view(output.shape[0], output.shape[1], 3, self.step_cols).permute(1, 2, 0, 3)
        output = output.contiguous().view(output.shape[0], 3, -1)
        return output[:, 1:], output[:, :1]


def load_horizonnet(variant: str = "mp3d", device: str = "cpu") -> HorizonNet:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, CHECKPOINTS[variant])
    # weights_only=True: never unpickle arbitrary objects from a downloaded checkpoint
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = HorizonNet(**ckpt.get("kwargs", {"backbone": "resnet50", "use_rnn": True}))
    net.load_state_dict(ckpt["state_dict"], strict=True)
    return net.to(device).eval()


@torch.inference_mode()
def predict(net: HorizonNet, rgb: np.ndarray, flip_tta: bool = True):
    """Equirect RGB uint8 (any 2:1 size) -> (y_bon (2, 1024), y_cor (1024,) logits), with the
    upstream horizontal-flip test-time augmentation."""
    import cv2

    img = cv2.resize(rgb, (1024, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = torch.from_numpy(img.transpose(2, 0, 1).copy())[None].to(next(net.parameters()).device)
    xs = torch.cat([x, torch.flip(x, [3])]) if flip_tta else x
    bon, cor = net(xs)
    if flip_tta:
        bon = (bon[0] + torch.flip(bon[1], [1])) / 2
        cor = (cor[0] + torch.flip(cor[1], [1])) / 2
    else:
        bon, cor = bon[0], cor[0]
    return bon.float().cpu().numpy(), cor.float().cpu().numpy().reshape(-1)
