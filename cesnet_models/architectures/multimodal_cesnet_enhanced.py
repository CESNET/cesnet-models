from enum import Enum
from functools import partial

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cesnet_models.architectures.multimodal_cesnet import (NormalizationEnum, conv_norm_from_enum,
                                                           linear_norm_from_enum)
from cesnet_models.constants import PPI_DIR_POS, PPI_IPT_POS, PPI_SIZE_POS


class GlobalPoolEnum(Enum):
    MAX = "max"
    AVG = "avg"
    def __str__(self): return self.value

class StemType(Enum):
    CONV = "conv"
    EMBED = "embed"
    EMBED_CONV = "embed-conv"
    NONE = "none"
    def __str__(self): return self.value

def init_weights_fn(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    # Calculate symmetric padding for a convolution
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class StdConv1d(nn.Conv1d):
    """Conv1d with Weight Standardization.

    Paper: Micro-Batch Training with Batch-Channel Normalization and Weight Standardization
        - https://arxiv.org/abs/1903.10520v2
    Implementation from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/std_conv.py
    """

    def __init__(
            self, in_channel, out_channels, kernel_size, stride=1, padding=None,
            dilation=1, groups=1, bias=False, eps=1e-6):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channel, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def forward(self, x):
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None,
            training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        x = F.conv1d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class BasicBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            act=None,
            conv=None,
            norm=None,
            dropout_rate_path=0.0,
            downsample_avg=False,
    ):
        super().__init__()
        act = act or nn.ReLU
        conv = conv or StdConv1d
        norm = norm or partial(nn.GroupNorm, num_groups=16)

        if stride > 1 or in_channels != out_channels:
            if downsample_avg:
                self.downsample = nn.Sequential(
                    nn.AvgPool1d(kernel_size=2, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                    conv(in_channels, out_channels, kernel_size=1, stride=1),
                    norm(out_channels),)
            else:
                self.downsample = nn.Sequential(
                    conv(in_channels, out_channels, kernel_size=1, stride=stride),
                    norm(out_channels),)
        else:
            self.downsample = None

        self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.norm1 = norm(out_channels)
        self.conv2 = conv(out_channels, out_channels, kernel_size=3)
        self.norm2 = norm(out_channels)
        self.drop_path = nn.Dropout(dropout_rate_path) if dropout_rate_path > 0 else nn.Identity()
        self.act = act(inplace=True)

    def forward(self, x):
        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        # residual
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = self.drop_path(out)
        out = self.act(out + shortcut)
        return out

class Bottleneck(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            act=None,
            conv=None,
            norm=None,
            dropout_rate_path=0.0,
            bottle_ratio=0.25,
            downsample_avg=False,
    ):
        super().__init__()
        act = act or nn.ReLU
        conv = conv or StdConv1d
        if norm is None:
            norm = lambda x: nn.GroupNorm(num_groups=16, num_channels=x)
        mid_channels = int(out_channels * bottle_ratio)

        if stride > 1 or in_channels != out_channels:
            if downsample_avg:
                self.downsample = nn.Sequential(
                    nn.AvgPool1d(kernel_size=2, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                    conv(in_channels, out_channels, kernel_size=1, stride=1),
                    norm(out_channels),)
            else:
                self.downsample = nn.Sequential(
                    conv(in_channels, out_channels, kernel_size=1, stride=stride),
                    norm(out_channels),)
        else:
            self.downsample = None

        self.conv1 = conv(in_channels, mid_channels, kernel_size=1)
        self.norm1 = norm(mid_channels)
        self.conv2 = conv(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride)
        self.norm2 = norm(mid_channels)
        self.conv3 = conv(mid_channels, out_channels, kernel_size=1)
        self.norm3 = norm(out_channels)
        self.drop_path = nn.Dropout(dropout_rate_path) if dropout_rate_path > 0 else nn.Identity()
        self.act = act(inplace=True)

    def forward(self, x):
        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        # residual
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.drop_path(out)
        out = self.act(out + shortcut)
        return out

def build_cnn_ppi(channels: tuple[int, ...],
                  strides: tuple[int, ...],
                  kernel_sizes: tuple[int, ...],
                  stem_output_channels: int,
                  conv, norm,
                  dropout_rate_path: float = 0.0,
                  downsample_avg=False,
                  block_class=None):
    assert len(channels) == len(strides) == len(kernel_sizes)
    if block_class is None:
        block_class = Bottleneck
    num_blocks = len(channels)
    blocks = []
    block_dprs = [x.item() for x in torch.linspace(0, dropout_rate_path, num_blocks)]
    for i in range(num_blocks):
        in_channels = stem_output_channels if i == 0 else channels[i - 1]
        if i == 0 and block_class == Bottleneck:
            kwargs = {"bottle_ratio": 0.5}
        else:
            kwargs = {}
        blocks.append(block_class(in_channels=in_channels,
                                  out_channels=channels[i],
                                  kernel_size=kernel_sizes[i],
                                  stride=strides[i],
                                  dropout_rate_path=block_dprs[i],
                                  downsample_avg=downsample_avg,
                                  conv=conv,norm=norm, **kwargs))
    return nn.Sequential(*blocks)

def build_cnn_ppi_stem(stem_type: StemType,
                       ppi_input_channels :int,
                       out_channels: int,
                       kernel_size: int,
                       packet_embedding_size: int,
                       packet_embedding_include_dirs: bool,
                       packet_embedding_init: bool,
                       conv, norm):
    stem = []
    packet_embedding = None
    stem_output_channels = ppi_input_channels
    if stem_type == StemType.CONV:
        stem = [
            conv(ppi_input_channels, out_channels, kernel_size=kernel_size, stride=1),
            norm(out_channels),
            nn.ReLU(inplace=True),
        ]
        stem_output_channels = out_channels
    if stem_type == StemType.EMBED or stem_type == StemType.EMBED_CONV:
        num_embeddings = 1500 * (2 if packet_embedding_include_dirs else 1) + 1
        packet_embedding = nn.Embedding(num_embeddings, packet_embedding_size, padding_idx=1500 if packet_embedding_include_dirs else 0)
        if packet_embedding_init:
            if packet_embedding_include_dirs:
                for i, size in enumerate(range(-1500, 1501)):
                    inital_embedding = torch.zeros(packet_embedding_size)
                    if size != 0:
                        inital_embedding[0] = 1 if size > 0 else 0
                        inital_embedding[1] = 1 if size < 0 else 0
                        inital_embedding[2] = abs(size) / 1500
                    packet_embedding.weight.data[i, :] = inital_embedding
            else:
                for i, size in enumerate(range(0, 1501)):
                    inital_embedding = torch.zeros(packet_embedding_size)
                    inital_embedding[0] = size / 1500
                    packet_embedding.weight.data[i, :] = inital_embedding
        conv1_input_channels = packet_embedding_size + (1 if packet_embedding_include_dirs else 2)
        if stem_type == StemType.EMBED_CONV:
            stem = [
                conv(conv1_input_channels, out_channels, kernel_size=kernel_size, stride=1),
                norm(out_channels),
                nn.ReLU(inplace=True),
            ]
            stem_output_channels = out_channels
        else:
            stem_output_channels = conv1_input_channels
    return nn.Sequential(*stem) if len(stem) > 0 else nn.Identity(), packet_embedding, stem_output_channels

class Multimodal_CESNET_Enhanced(nn.Module):
    def __init__(self, num_classes: int,
                       flowstats_input_size: int,
                       ppi_input_channels: int,
                       use_flowstats: bool = True,
                       init_weights: bool = True,
                       cnn_ppi_stem_type: StemType = StemType.EMBED_CONV, packet_embedding_size: int = 7, packet_embedding_include_dirs: bool = True, packet_embedding_init: bool = True,
                       conv_normalization: NormalizationEnum = NormalizationEnum.BATCH_NORM, linear_normalization: NormalizationEnum = NormalizationEnum.BATCH_NORM, group_norm_groups: int = 16,
                       cnn_ppi_channels: tuple[int, ...] = (128, 256, 384, 384), cnn_ppi_strides: tuple[int, ...] = (1, 1, 2, 1), cnn_ppi_kernel_sizes: tuple[int, ...] = (7, 5, 5, 3),
                       cnn_ppi_use_stdconv: bool = True, cnn_ppi_downsample_avg: bool = True, cnn_ppi_blocks_dropout_rate: float = 0.0,
                       cnn_ppi_global_pool: GlobalPoolEnum = GlobalPoolEnum.AVG, cnn_ppi_dropout_rate: float = 0.0,
                       mlp_flowstats_size1: int = 256, mlp_flowstats_size2: int = 64, mlp_flowstats_num_hidden: int = 1, mlp_flowstats_dropout_rate: float = 0.0,
                       mlp_shared_size: int = 512, mlp_shared_dropout_rate: float = 0.0,
                       ):
        super().__init__()
        assert ppi_input_channels == 3
        self.num_classes = num_classes
        self.flowstats_input_size = flowstats_input_size
        self.use_flowstats = use_flowstats
        self.packet_embedding_include_dirs = packet_embedding_include_dirs
        self.mlp_shared_size = mlp_shared_size
        mlp_shared_input_size = cnn_ppi_channels[-1] + mlp_flowstats_size2 if use_flowstats else cnn_ppi_channels[-1]
        conv = StdConv1d if cnn_ppi_use_stdconv else partial(nn.Conv1d, bias=False)
        conv_norm = partial(conv_norm_from_enum, norm_enum=conv_normalization, group_norm_groups=group_norm_groups)
        linear_norm = partial(linear_norm_from_enum, norm_enum=linear_normalization)
        self.cnn_ppi_stem_type = cnn_ppi_stem_type
        self.cnn_ppi_stem, self.packet_embedding, stem_output_channels = build_cnn_ppi_stem(stem_type=cnn_ppi_stem_type,
                                                                                                  ppi_input_channels=ppi_input_channels,
                                                                                                  out_channels=cnn_ppi_channels[0] // 2,
                                                                                                  kernel_size=7,
                                                                                                  packet_embedding_size=packet_embedding_size,
                                                                                                  packet_embedding_include_dirs=packet_embedding_include_dirs,
                                                                                                  packet_embedding_init=packet_embedding_init,
                                                                                                  conv=conv, norm=conv_norm)
        self.cnn_ppi = build_cnn_ppi(channels=cnn_ppi_channels,
                                     strides=cnn_ppi_strides,
                                     kernel_sizes=cnn_ppi_kernel_sizes,
                                     stem_output_channels=stem_output_channels,
                                     dropout_rate_path=cnn_ppi_blocks_dropout_rate,
                                     downsample_avg=cnn_ppi_downsample_avg,
                                     conv=conv, norm=conv_norm)
        self.cnn_global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1) if cnn_ppi_global_pool == GlobalPoolEnum.AVG else nn.AdaptiveMaxPool1d(output_size=1),
            nn.Flatten(),
            nn.Dropout(cnn_ppi_dropout_rate) if cnn_ppi_dropout_rate > 0 else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.mlp_flowstats = nn.Sequential(
            nn.Linear(flowstats_input_size, mlp_flowstats_size1),
            linear_norm(mlp_flowstats_size1),
            nn.ReLU(inplace=True),

            *(nn.Sequential(
                nn.Linear(mlp_flowstats_size1, mlp_flowstats_size1),
                linear_norm(mlp_flowstats_size1),
                nn.ReLU(inplace=True),) for _ in range(mlp_flowstats_num_hidden)),

            nn.Linear(mlp_flowstats_size1, mlp_flowstats_size2),
            linear_norm(mlp_flowstats_size2),
            nn.Dropout(mlp_flowstats_dropout_rate) if mlp_flowstats_dropout_rate > 0 else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.mlp_shared = nn.Sequential(
            nn.Linear(mlp_shared_input_size, mlp_shared_size),
            linear_norm(mlp_shared_size),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_shared_dropout_rate) if mlp_shared_dropout_rate > 0 else nn.Identity(),
        )
        self.classifier = nn.Linear(self.mlp_shared_size, num_classes)
        if init_weights:
            self.apply(init_weights_fn)

    def forward_features(self, ppi, flowstats):
        if self.cnn_ppi_stem_type == StemType.EMBED or self.cnn_ppi_stem_type == StemType.EMBED_CONV:
            assert self.packet_embedding is not None
            if self.packet_embedding_include_dirs:
                embedding_input = (ppi[:, PPI_SIZE_POS, :] * ppi[:, PPI_DIR_POS, :]).int() + 1500
                ppi_embedded = torch.cat((
                    ppi[:, PPI_IPT_POS, :].unsqueeze(-1),
                    self.packet_embedding(embedding_input)
                ), dim=2).transpose(1, 2)
            else:
                embedding_input = ppi[:, PPI_SIZE_POS, :].int()
                ppi_embedded = torch.cat((
                    ppi[:, PPI_IPT_POS, :].unsqueeze(-1),
                    ppi[:, PPI_DIR_POS, :].unsqueeze(-1),
                    self.packet_embedding(embedding_input)
                ), dim=2).transpose(1, 2)
        else:
            ppi_embedded = ppi
        out = self.cnn_ppi_stem(ppi_embedded)
        out = self.cnn_ppi(out)
        out = self.cnn_global_pooling(out)
        if self.use_flowstats:
            out_flowstats = self.mlp_flowstats(flowstats)
            out = torch.column_stack([out, out_flowstats])
        out = self.mlp_shared(out)
        return out

    def forward_head(self, x):
        return self.classifier(x)

    def forward(self, *x: tuple) -> Tensor:
        if len(x) == 1:
            x = x[0]
        ppi, flowstats = x
        out = self.forward_features(ppi=ppi, flowstats=flowstats)
        out = self.forward_head(out)
        return out
