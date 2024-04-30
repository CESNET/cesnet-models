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

class Multimodal_CESNET_Enhanced(nn.Module):
    def __init__(self, num_classes: int,
                       flowstats_input_size: int,
                       ppi_input_channels: int,
                       use_flowstats: bool = True,
                       init_weights: bool = True,
                       packet_embedding_size: int = 4, packet_embedding_init: bool = False, packet_embedding_include_dirs: bool = False,
                       conv_normalization: NormalizationEnum = NormalizationEnum.GROUP_NORM, linear_normalization: NormalizationEnum = NormalizationEnum.LAYER_NORM, group_norm_groups: int = 32,
                       cnn_ppi_channels1: int = 128, cnn_ppi_channels2: int = 256, cnn_ppi_channels3: int = 256, cnn_ppi_num_blocks: int = 3,
                       cnn_ppi_global_pooling: GlobalPoolEnum = GlobalPoolEnum.AVG, cnn_ppi_dropout_rate: float = 0.1, use_standardized_conv: bool = True,
                       mlp_flowstats_size1: int = 128, mlp_flowstats_size2: int = 64, mlp_flowstats_num_hidden: int = 2, mlp_flowstats_dropout_rate: float = 0.1,
                       mlp_shared_size: int = 512, mlp_shared_num_hidden: int = 0, mlp_shared_dropout_rate: float = 0.2,
                       ):
        super().__init__()
        assert ppi_input_channels == 3
        self.num_classes = num_classes
        self.flowstats_input_size = flowstats_input_size
        self.use_flowstats = use_flowstats
        self.packet_embedding_size = packet_embedding_size
        self.packet_embedding_include_dirs = packet_embedding_include_dirs
        self.mlp_shared_size = mlp_shared_size

        CNN_PPI_OUTPUT_LEN = 10
        conv = StdConv1d if use_standardized_conv else partial(nn.Conv1d, bias=False)
        conv_norm = partial(conv_norm_from_enum, norm_enum=conv_normalization, group_norm_groups=group_norm_groups)
        linear_norm = partial(linear_norm_from_enum, norm_enum=linear_normalization)
        mlp_shared_input_size = cnn_ppi_channels3 + mlp_flowstats_size2 if use_flowstats else cnn_ppi_channels3

        if self.packet_embedding_size > 0:
            num_embeddings = 1500 * (2 if packet_embedding_include_dirs else 1) + 1
            self.packet_sizes_embedding = nn.Embedding(num_embeddings, packet_embedding_size, padding_idx=1500 if packet_embedding_include_dirs else 0)
            if packet_embedding_init:
                if packet_embedding_include_dirs:
                    for i, size in enumerate(range(-1500, 1501)):
                        inital_embedding = torch.zeros(packet_embedding_size)
                        if size != 0:
                            inital_embedding[0] = -1 if size < 0 else 1
                            inital_embedding[1] = abs(size) / 1500
                        self.packet_sizes_embedding.weight.data[i, :] = inital_embedding
                else:
                    for i, size in enumerate(range(0, 1501)):
                        inital_embedding = torch.zeros(packet_embedding_size)
                        inital_embedding[0] = size / 1500
                        self.packet_sizes_embedding.weight.data[i, :] = inital_embedding
        conv1_input_channels = ppi_input_channels if packet_embedding_size == 0 else packet_embedding_size + (1 if packet_embedding_include_dirs else 2)

        self.cnn_ppi = nn.Sequential(
            # [(W−K+2P)/S]+1
            # Input: 30 * 3
            conv(conv1_input_channels, cnn_ppi_channels1, kernel_size=7, stride=1, padding=3),
            *conv_norm(cnn_ppi_channels1),
            nn.ReLU(inplace=False),

            # 30 x channels1
            *(nn.Sequential(
                conv(cnn_ppi_channels1, cnn_ppi_channels1, kernel_size=5, stride=1, padding=2),
                *conv_norm(cnn_ppi_channels1),) for _ in range(cnn_ppi_num_blocks)),
                nn.ReLU(inplace=False),

            # 30 x channels1
            conv(cnn_ppi_channels1, cnn_ppi_channels2, kernel_size=5, stride=2),
            *conv_norm(cnn_ppi_channels2),
            nn.ReLU(inplace=False),
            # 15 * channels2
            conv(cnn_ppi_channels2, cnn_ppi_channels2, kernel_size=4, stride=1),
            *conv_norm(cnn_ppi_channels2),
            nn.ReLU(inplace=False),
            # 12 * channels2
            conv(cnn_ppi_channels2, cnn_ppi_channels3, kernel_size=3, stride=1),
            *conv_norm(cnn_ppi_channels3),
            nn.ReLU(inplace=False),
            # 10 * channels3
            # CNN_PPI_OUTPUT_LEN = 10
        )
        self.cnn_global_pooling = nn.Sequential(
            nn.AvgPool1d(kernel_size=CNN_PPI_OUTPUT_LEN) if cnn_ppi_global_pooling == GlobalPoolEnum.AVG else nn.MaxPool1d(kernel_size=CNN_PPI_OUTPUT_LEN),
            nn.Dropout(cnn_ppi_dropout_rate),
            nn.Flatten(),
        )
        self.mlp_flowstats = nn.Sequential(
            nn.Linear(flowstats_input_size, mlp_flowstats_size1),
            *linear_norm(mlp_flowstats_size1),
            nn.ReLU(inplace=False),

            *(nn.Sequential(
                nn.Linear(mlp_flowstats_size1, mlp_flowstats_size1),
                *linear_norm(mlp_flowstats_size1)) for _ in range(mlp_flowstats_num_hidden)),
                nn.ReLU(inplace=False),

            nn.Linear(mlp_flowstats_size1, mlp_flowstats_size2),
            *linear_norm(mlp_flowstats_size2),
            nn.Dropout(mlp_flowstats_dropout_rate),
            # nn.ReLU(inplace=False), # To have inputs for mlp_shared from both modalities without ReLU
        )
        self.mlp_shared = nn.Sequential(
            nn.Linear(mlp_shared_input_size, mlp_shared_size),
            *linear_norm(mlp_shared_size),
            nn.Dropout(mlp_shared_dropout_rate),
            nn.ReLU(inplace=False),

            *(nn.Sequential(
                nn.Linear(mlp_shared_size, mlp_shared_size),
                *linear_norm(mlp_shared_size),
                nn.Dropout(mlp_shared_dropout_rate),
                nn.ReLU(inplace=False)) for _ in range(mlp_shared_num_hidden)),
        )
        self.classifier = nn.Linear(mlp_shared_size, num_classes)
        if init_weights:
            self.apply(init_weights_fn)

    def _forward_impl(self, ppi, flowstats):
        if self.packet_embedding_size > 0:
            if self.packet_embedding_include_dirs:
                embedding_input = (ppi[:, PPI_SIZE_POS, :] * ppi[:, PPI_DIR_POS, :]).int() + 1500
                ppi_embedded = torch.cat((
                    ppi[:, PPI_IPT_POS, :].unsqueeze(-1),
                    self.packet_sizes_embedding(embedding_input)
                ), dim=2).transpose(1, 2)
            else:
                embedding_input = ppi[:, PPI_SIZE_POS, :].int()
                ppi_embedded = torch.cat((
                    ppi[:, PPI_IPT_POS, :].unsqueeze(-1),
                    ppi[:, PPI_DIR_POS, :].unsqueeze(-1),
                    self.packet_sizes_embedding(embedding_input)
                ), dim=2).transpose(1, 2)
        else:
            ppi_embedded = ppi
        out = self.cnn_ppi(ppi_embedded)
        out = self.cnn_global_pooling(out)
        if self.use_flowstats:
            out_flowstats = self.mlp_flowstats(flowstats)
            out = torch.column_stack([out, out_flowstats])
        out = self.mlp_shared(out)
        logits = self.classifier(out)
        return logits

    def forward(self, *x: tuple) -> Tensor:
        if len(x) == 1:
            x = x[0]
        ppi, flowstats = x
        return self._forward_impl(ppi=ppi, flowstats=flowstats)
