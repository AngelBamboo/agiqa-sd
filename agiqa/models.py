"""Quality regressors used in the study.

The backbone is deliberately small: the paper targets a realistic single-GPU
setting, and a compact model also makes the shortcut diagnosis cleaner — if a
2.9 M-parameter network can read generator identity, the cue is in the data,
not in model capacity.
"""
import torch
import torch.nn as nn
import torchvision


class SmallBackbone(nn.Module):
    """ResNet-18 (optionally ImageNet-initialised) with a configurable head."""

    def __init__(self, pretrained=True, width=512):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 512

    def forward(self, x):
        f = self.stem(x)
        return self.pool(f).flatten(1)


class QualityNet(nn.Module):
    """Backbone + optional generator head for adversarial invariance."""

    def __init__(self, pretrained=True, num_generators=6, dropout=0.2):
        super().__init__()
        self.backbone = SmallBackbone(pretrained=pretrained)
        d = self.backbone.feat_dim
        self.quality = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        # The generator head is detached-free: the gradient reversal layer turns
        # its gradient into a feature subtraction when `grl_lambda` > 0.
        self.gen_head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(True),
                                      nn.Linear(128, num_generators))
        self.grl_lambda = 0.0

    def forward(self, x, return_logits=False):
        f = self.backbone(x)
        q = self.quality(f).squeeze(1)
        if not return_logits:
            return q
        g = self.gen_head(GradReverse.apply(f, self.grl_lambda))
        return q, g


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
