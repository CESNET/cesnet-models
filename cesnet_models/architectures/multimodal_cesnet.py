from enum import Enum
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NormalizationEnum(Enum):
    BATCH_NORM = "batch-norm"
    LAYER_NORM = "layer-norm"
    INSTANCE_NORM = "instance-norm"
    NO_NORM = "no-norm"
    def __str__(self): return self.value

def conv_norm_from_enum(size: int, norm_enum: NormalizationEnum):
    if norm_enum == NormalizationEnum.BATCH_NORM:
        return [nn.BatchNorm1d(size)]
    elif norm_enum == NormalizationEnum.INSTANCE_NORM:
        return [nn.InstanceNorm1d(size)]
    elif norm_enum == NormalizationEnum.NO_NORM:
        return []
    else:
        raise ValueError(f"Bad normalization for nn.Conv1d: {str(norm_enum)}")

def linear_norm_from_enum(size: int, norm_enum: NormalizationEnum):
    if norm_enum == NormalizationEnum.BATCH_NORM:
        return [nn.BatchNorm1d(size)]
    if norm_enum == NormalizationEnum.LAYER_NORM:
        return [nn.LayerNorm(size)]
    elif norm_enum == NormalizationEnum.NO_NORM:
        return []
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
        return self.gem(x, p=self.p, eps=self.eps)

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
                       use_flowstats: bool = True, add_ppi_to_flowstats: bool = False,
                       conv_normalization: NormalizationEnum = NormalizationEnum.BATCH_NORM, linear_normalization: NormalizationEnum = NormalizationEnum.BATCH_NORM,
                       cnn_channels1: int = 200, cnn_channels2: int = 300, cnn_channels3: int = 300, cnn_num_hidden: int = 3, cnn_depthwise: bool = False, cnn_pooling_dropout_rate: float = 0.1,
                       flowstats_size: int = 225, flowstats_out_size: int = 225, flowstats_num_hidden: int = 2, flowstats_dropout_rate: float = 0.1,
                       latent_size: int = 600, latent_num_hidden: int = 0, latent_dropout_rate: float = 0.2,
                       ):
        super().__init__()
        if add_ppi_to_flowstats and not use_flowstats:
            raise ValueError("add_ppi_to_flowstats requires use_flowstats")
        if cnn_depthwise and cnn_channels1 % ppi_input_channels != 0:
            raise ValueError(f"cnn_channels1 ({cnn_channels1}) must be divisible by ppi_input_channels ({ppi_input_channels}) when using cnn_depthwise")

        self.num_classes = num_classes
        self.flowstats_input_size = flowstats_input_size
        self.ppi_input_channels = ppi_input_channels
        self.use_flowstats = use_flowstats
        self.add_ppi_to_flowstats = add_ppi_to_flowstats
        self.latent_size = latent_size

        CNN_PPI_OUTPUT_LEN = 10
        PPI_LEN = 30
        conv_norm = partial(conv_norm_from_enum, norm_enum=conv_normalization)
        linear_norm = partial(linear_norm_from_enum, norm_enum=linear_normalization)
        conv1d_groups = ppi_input_channels if cnn_depthwise else 1
        mlp_flowstats_input_size = flowstats_input_size + (ppi_input_channels * PPI_LEN) if add_ppi_to_flowstats else flowstats_input_size
        mlp_shared_input_size =  cnn_channels3 + flowstats_out_size if use_flowstats else cnn_channels3

        self.cnn_ppi = nn.Sequential(
            # [(W−K+2P)/S]+1
            # Input: 30 * 3
            nn.Conv1d(self.ppi_input_channels, cnn_channels1, kernel_size=7, stride=1, groups=conv1d_groups, padding=3),
            nn.ReLU(inplace=False),
            *conv_norm(cnn_channels1),

            # 30 x channels1
            *(nn.Sequential(
                nn.Conv1d(cnn_channels1, cnn_channels1, kernel_size=5, stride=1, groups=conv1d_groups, padding=2),
                nn.ReLU(inplace=False),
                *conv_norm(cnn_channels1),) for _ in range(cnn_num_hidden)),

            # 30 x channels1
            nn.Conv1d(cnn_channels1, cnn_channels2, kernel_size=5, stride=1),
            nn.ReLU(inplace=False),
            *conv_norm(cnn_channels2),
            # 26 * channels2
            nn.Conv1d(cnn_channels2, cnn_channels2, kernel_size=5, stride=1),
            nn.ReLU(inplace=False),
            *conv_norm(cnn_channels2),
            # 22 * channels2
            nn.Conv1d(cnn_channels2, cnn_channels3, kernel_size=4, stride=2),
            nn.ReLU(inplace=False),
            # 10 * channels3
            # CNN_PPI_OUTPUT_LEN = 10
        )
        self.cnn_global_pooling = nn.Sequential(
            GeM(kernel_size=CNN_PPI_OUTPUT_LEN),
            nn.Flatten(),
            *linear_norm(cnn_channels3),
            nn.Dropout(cnn_pooling_dropout_rate),
        )
        self.mlp_flowstats = nn.Sequential(
            nn.Linear(mlp_flowstats_input_size, flowstats_size),
            nn.ReLU(inplace=False),
            *linear_norm(flowstats_size),

            *(nn.Sequential(
                nn.Linear(flowstats_size, flowstats_size),
                nn.ReLU(inplace=False),
                *linear_norm(flowstats_size)) for _ in range(flowstats_num_hidden)),

            nn.Linear(flowstats_size, flowstats_out_size),
            nn.ReLU(inplace=False),
            *linear_norm(flowstats_out_size),
            nn.Dropout(flowstats_dropout_rate),
        )
        self.mlp_shared = nn.Sequential(
            nn.Linear(mlp_shared_input_size, latent_size),
            nn.ReLU(inplace=False),
            *linear_norm(latent_size),
            nn.Dropout(latent_dropout_rate),

            *(nn.Sequential(
                nn.Linear(latent_size, latent_size),
                nn.ReLU(inplace=False),
                *linear_norm(latent_size),
                nn.Dropout(latent_dropout_rate)) for _ in range(latent_num_hidden)),
        )
        self.out = nn.Linear(latent_size, num_classes)

    def _forward_impl(self, ppi, flowstats):
        out = self.cnn_ppi(ppi)
        out = self.cnn_global_pooling(out)
        if self.use_flowstats:
            if self.add_ppi_to_flowstats:
                flowstats_input = torch.column_stack([torch.flatten(ppi, 1), flowstats])
            else:
                flowstats_input = flowstats
            out_flowstats = self.mlp_flowstats(flowstats_input)
            out = torch.column_stack([out, out_flowstats])
        out = self.mlp_shared(out)
        logits = self.out(out)
        return logits

    def forward(self, *x: tuple) -> Tensor:
        if len(x) == 1:
            x = x[0]
        ppi, flowstats = x
        return self._forward_impl(ppi=ppi, flowstats=flowstats)
