from enum import Enum
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NormalizationEnum(Enum):
    BATCH_NORM = "batch-norm"
    GROUP_NORM = "group-norm"
    LAYER_NORM = "layer-norm"
    INSTANCE_NORM = "instance-norm"
    NO_NORM = "no-norm"
    def __str__(self): return self.value

def conv_norm_from_enum(size: int, norm_enum: NormalizationEnum, group_norm_groups: int = 16):
    if norm_enum == NormalizationEnum.BATCH_NORM:
        return nn.BatchNorm1d(size)
    elif norm_enum == NormalizationEnum.GROUP_NORM:
        return nn.GroupNorm(num_channels=size, num_groups=group_norm_groups)
    elif norm_enum == NormalizationEnum.INSTANCE_NORM:
        return nn.InstanceNorm1d(size)
    elif norm_enum == NormalizationEnum.NO_NORM:
        return nn.Identity()
    else:
        raise ValueError(f"Bad normalization for nn.Conv1d: {str(norm_enum)}")

def linear_norm_from_enum(size: int, norm_enum: NormalizationEnum):
    if norm_enum == NormalizationEnum.BATCH_NORM:
        return nn.BatchNorm1d(size)
    if norm_enum == NormalizationEnum.LAYER_NORM:
        return nn.LayerNorm(size)
    elif norm_enum == NormalizationEnum.NO_NORM:
        return nn.Identity()
    else:
        raise ValueError(f"Bad normalization for nn.Linear: {str(norm_enum)}")

class GeM(nn.Module):
    """
    https://www.kaggle.com/code/scaomath/g2net-1d-cnn-gem-pool-pytorch-train-inference
    """
    def __init__(self, kernel_size, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps) # type: ignore

    def gem(self, x, p: int | nn.Parameter = 3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p), self.kernel_size).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + \
                "(" + "kernel_size=" + str(self.kernel_size) + ", p=" + "{:.4f}".format(self.p.data.tolist()[0]) + \
                ", eps=" + str(self.eps) + ")"

class Multimodal_CESNET(nn.Module):
    def __init__(self, num_classes: int,
                       flowstats_input_size: int,
                       ppi_input_channels: int,
                       use_flowstats: bool = True, add_ppi_to_mlp_flowstats: bool = False,
                       conv_normalization: NormalizationEnum = NormalizationEnum.BATCH_NORM, linear_normalization: NormalizationEnum = NormalizationEnum.BATCH_NORM,
                       cnn_ppi_channels1: int = 200, cnn_ppi_channels2: int = 300, cnn_ppi_channels3: int = 300, cnn_ppi_num_blocks: int = 3, cnn_ppi_depthwise: bool = False,
                       cnn_ppi_use_pooling: bool = True, cnn_ppi_dropout_rate: float = 0.1,
                       mlp_flowstats_size1: int = 225, mlp_flowstats_size2: int = 225, mlp_flowstats_num_hidden: int = 2, mlp_flowstats_dropout_rate: float = 0.1,
                       mlp_shared_size: int = 600, mlp_shared_num_hidden: int = 0, mlp_shared_dropout_rate: float = 0.2,
                       ):
        super().__init__()
        if add_ppi_to_mlp_flowstats and not use_flowstats:
            raise ValueError("add_ppi_to_mlp_flowstats requires use_flowstats")
        if cnn_ppi_depthwise and cnn_ppi_channels1 % ppi_input_channels != 0:
            raise ValueError(f"cnn_ppi_channels1 ({cnn_ppi_channels1}) must be divisible by ppi_input_channels ({ppi_input_channels}) when using cnn_ppi_depthwise")

        self.num_classes = num_classes
        self.flowstats_input_size = flowstats_input_size
        self.ppi_input_channels = ppi_input_channels
        self.use_flowstats = use_flowstats
        self.add_ppi_to_mlp_flowstats = add_ppi_to_mlp_flowstats
        self.mlp_shared_size = mlp_shared_size
        self.cnn_ppi_use_pooling = cnn_ppi_use_pooling

        CNN_PPI_OUTPUT_LEN = 10
        PPI_LEN = 30
        conv_norm = partial(conv_norm_from_enum, norm_enum=conv_normalization)
        linear_norm = partial(linear_norm_from_enum, norm_enum=linear_normalization)
        conv1d_groups = ppi_input_channels if cnn_ppi_depthwise else 1
        mlp_flowstats_input_size = flowstats_input_size + (ppi_input_channels * PPI_LEN) if add_ppi_to_mlp_flowstats else flowstats_input_size
        mlp_shared_input_size = mlp_flowstats_size2 if use_flowstats else 0
        if cnn_ppi_use_pooling:
            mlp_shared_input_size += cnn_ppi_channels3
        else:
            mlp_shared_input_size += cnn_ppi_channels3 * CNN_PPI_OUTPUT_LEN

        self.cnn_ppi = nn.Sequential(
            # [(W−K+2P)/S]+1
            # Input: 30 * 3
            nn.Conv1d(self.ppi_input_channels, cnn_ppi_channels1, kernel_size=7, stride=1, groups=conv1d_groups, padding=3),
            nn.ReLU(inplace=False),
            conv_norm(cnn_ppi_channels1),

            # 30 x channels1
            *(nn.Sequential(
                nn.Conv1d(cnn_ppi_channels1, cnn_ppi_channels1, kernel_size=5, stride=1, groups=conv1d_groups, padding=2),
                nn.ReLU(inplace=False),
                conv_norm(cnn_ppi_channels1),) for _ in range(cnn_ppi_num_blocks)),

            # 30 x channels1
            nn.Conv1d(cnn_ppi_channels1, cnn_ppi_channels2, kernel_size=5, stride=1),
            nn.ReLU(inplace=False),
            conv_norm(cnn_ppi_channels2),
            # 26 * channels2
            nn.Conv1d(cnn_ppi_channels2, cnn_ppi_channels2, kernel_size=5, stride=1),
            nn.ReLU(inplace=False),
            conv_norm(cnn_ppi_channels2),
            # 22 * channels2
            nn.Conv1d(cnn_ppi_channels2, cnn_ppi_channels3, kernel_size=4, stride=2),
            nn.ReLU(inplace=False),
            # 10 * channels3
            # CNN_PPI_OUTPUT_LEN = 10
        )
        if cnn_ppi_use_pooling:
            self.cnn_global_pooling = nn.Sequential(
                GeM(kernel_size=CNN_PPI_OUTPUT_LEN),
                nn.Flatten(),
                linear_norm(cnn_ppi_channels3),
                nn.Dropout(cnn_ppi_dropout_rate),
            )
        else:
            self.cnn_flatten_without_pooling = nn.Sequential(
                nn.Flatten(),
                linear_norm(cnn_ppi_channels3 * CNN_PPI_OUTPUT_LEN),
                nn.Dropout(cnn_ppi_dropout_rate),
            )
        self.mlp_flowstats = nn.Sequential(
            nn.Linear(mlp_flowstats_input_size, mlp_flowstats_size1),
            nn.ReLU(inplace=False),
            linear_norm(mlp_flowstats_size1),

            *(nn.Sequential(
                nn.Linear(mlp_flowstats_size1, mlp_flowstats_size1),
                nn.ReLU(inplace=False),
                linear_norm(mlp_flowstats_size1)) for _ in range(mlp_flowstats_num_hidden)),

            nn.Linear(mlp_flowstats_size1, mlp_flowstats_size2),
            nn.ReLU(inplace=False),
            linear_norm(mlp_flowstats_size2),
            nn.Dropout(mlp_flowstats_dropout_rate),
        )
        self.mlp_shared = nn.Sequential(
            nn.Linear(mlp_shared_input_size, mlp_shared_size),
            nn.ReLU(inplace=False),
            linear_norm(mlp_shared_size),
            nn.Dropout(mlp_shared_dropout_rate),

            *(nn.Sequential(
                nn.Linear(mlp_shared_size, mlp_shared_size),
                nn.ReLU(inplace=False),
                linear_norm(mlp_shared_size),
                nn.Dropout(mlp_shared_dropout_rate)) for _ in range(mlp_shared_num_hidden)),
        )
        self.classifier = nn.Linear(mlp_shared_size, num_classes)

    def forward_features(self, ppi, flowstats):
        out = self.cnn_ppi(ppi)
        if self.cnn_ppi_use_pooling:
            out = self.cnn_global_pooling(out)
        else:
            out = self.cnn_flatten_without_pooling(out)
        if self.use_flowstats:
            if self.add_ppi_to_mlp_flowstats:
                flowstats_input = torch.column_stack([torch.flatten(ppi, 1), flowstats])
            else:
                flowstats_input = flowstats
            out_flowstats = self.mlp_flowstats(flowstats_input)
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
