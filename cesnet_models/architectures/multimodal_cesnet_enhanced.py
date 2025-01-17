from enum import Enum
from functools import partial

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cesnet_models.architectures.multimodal_cesnet import (NormalizationEnum, conv_norm_from_enum,
                                                           linear_norm_from_enum)
from cesnet_models.constants import PPI_DIR_POS, PPI_IPT_POS, PPI_SIZE_POS
from cesnet_models.helpers import convert_str_to_enum


class PacketSizeInitEnum(Enum):
    RANDOM = "random"
    BASIC = "basic"
    SAME = "same"
    WITH_DIR = "with-dir"
    PLE = "ple"
    def __str__(self): return self.value

class ProcessIPT(Enum):
    NONE = "none"
    DIRECT = "direct"
    EMBED = "embed"
    def __str__(self): return self.value

class GlobalPoolEnum(Enum):
    MAX = "max"
    AVG = "avg"
    GEM_3 = "gem-3"
    GEM_6 = "gem-6"
    GEM_3_LEARNABLE = "gem-3-learnable"
    GEM_6_LEARNABLE = "gem-6-learnable"
    def __str__(self): return self.value

class StemType(Enum):
    CONV = "conv"
    EMBED = "embed"
    EMBED_CONV = "embed-conv"
    NONE = "none"
    def __str__(self): return self.value

class DropoutType(Enum):
    REGULAR = "regular"
    CHANNELS = "channels"
    def __str__(self): return self.value

def init_weights_fn(module: nn.Module):
    # Changed to zero init for bias, TODO experiment with .weight
    if isinstance(module, nn.Linear):
        # nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        # nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    # Calculate symmetric padding for a convolution
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class AdaptiveGeM(nn.Module):
    def __init__(self, output_size: int, p: float = 3.0, eps: float = 1e-6, learnable_p: bool = False):
        super().__init__()
        if learnable_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool1d(x.clamp(min=self.eps).pow(self.p), output_size=self.output_size).pow(1./self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item() if isinstance(self.p, nn.Parameter) else self.p:.4f}, eps={self.eps})"

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

class PadConv1d(nn.Conv1d):
    """Conv1d with automatic padding calculation.
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
        x = F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
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
            dropout_type=DropoutType.REGULAR,
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
        self.act = act(inplace=True)
        if dropout_rate_path > 0:
            if dropout_type == DropoutType.REGULAR:
                self.drop_path = nn.Dropout(dropout_rate_path)
            elif dropout_type == DropoutType.CHANNELS:
                self.drop_path = nn.Dropout1d(dropout_rate_path)
        else:
            self.drop_path = nn.Identity()

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
            dropout_type=DropoutType.REGULAR,
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
        self.act = act(inplace=True)
        if dropout_rate_path > 0:
            if dropout_type == DropoutType.REGULAR:
                self.drop_path = nn.Dropout(dropout_rate_path)
            elif dropout_type == DropoutType.CHANNELS:
                self.drop_path = nn.Dropout1d(dropout_rate_path)
        else:
            self.drop_path = nn.Identity()

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
                  dropout_type: DropoutType = DropoutType.REGULAR,
                  downsample_avg=False,
                  block_class=None,
                  first_block_bottle_ratio: float = 0.5,):
    assert len(channels) == len(strides) == len(kernel_sizes)
    if block_class is None:
        block_class = Bottleneck
    num_blocks = len(channels)
    blocks = []
    block_dprs = [x.item() for x in torch.linspace(0, dropout_rate_path, num_blocks)]
    for i in range(num_blocks):
        in_channels = stem_output_channels if i == 0 else channels[i - 1]
        if i == 0 and block_class == Bottleneck:
            kwargs = {"bottle_ratio": first_block_bottle_ratio}
        else:
            kwargs = {}
        blocks.append(block_class(in_channels=in_channels,
                                  out_channels=channels[i],
                                  kernel_size=kernel_sizes[i],
                                  stride=strides[i],
                                  dropout_rate_path=block_dprs[i],
                                  dropout_type=dropout_type,
                                  downsample_avg=downsample_avg,
                                  conv=conv,norm=norm, **kwargs))
    return nn.Sequential(*blocks)

def build_cnn_ppi_stem(stem_type: StemType,
                       ppi_input_channels :int,
                       out_channels: int,
                       kernel_size: int,
                       conv,
                       norm,
                       pe_size_embedding: int,
                       pe_size_include_dir: bool,
                       pe_size_init: PacketSizeInitEnum,
                       pe_size_ple_bin_size: int,
                       pe_ipt_embedding: int,
                       pe_ipt_processing: ProcessIPT,
                       pe_onehot_dirs: bool,
):
    stem = []
    packet_size_nn_embedding = None
    packet_ipt_nn_embedding = None
    ipt_bins = None
    stem_output_channels = ppi_input_channels
    if stem_type == StemType.CONV:
        stem = [
            conv(ppi_input_channels, out_channels, kernel_size=kernel_size, stride=1),
            norm(out_channels),
            nn.ReLU(inplace=True),
        ]
        stem_output_channels = out_channels
    if stem_type == StemType.EMBED or stem_type == StemType.EMBED_CONV:
        # Packet size embedding
        packet_size_nn_embedding = nn.Embedding(num_embeddings=1500 * (2 if pe_size_include_dir else 1) + 1, embedding_dim=pe_size_embedding, padding_idx=1500 if pe_size_include_dir else 0)
        ple_size_bins = list(range(0, 1500, pe_size_ple_bin_size))
        if pe_size_init != PacketSizeInitEnum.RANDOM:
            if pe_size_include_dir:
                for i, size in enumerate(range(-1500, 1501)):
                    if pe_size_init == PacketSizeInitEnum.BASIC:
                        inital_embedding = torch.zeros(pe_size_embedding)
                        inital_embedding[0] = abs(size) / 1500
                    elif pe_size_init == PacketSizeInitEnum.SAME:
                        inital_embedding = torch.ones(pe_size_embedding) * (abs(size) / 1500)
                    elif pe_size_init == PacketSizeInitEnum.WITH_DIR:
                        inital_embedding = torch.zeros(pe_size_embedding)
                        inital_embedding[0] = abs(size) / 1500
                        if size != 0:
                            inital_embedding[1] = 1 if size > 0 else 0
                            inital_embedding[2] = 1 if size < 0 else 0
                    packet_size_nn_embedding.weight.data[i, :] = inital_embedding
            else:
                for i, size in enumerate(range(0, 1501)):
                    if pe_size_init == PacketSizeInitEnum.BASIC:
                        inital_embedding = torch.zeros(pe_size_embedding)
                        inital_embedding[0] = size / 1500
                    elif pe_size_init == PacketSizeInitEnum.SAME:
                        inital_embedding = torch.ones(pe_size_embedding) * (size / 1500)
                    elif pe_size_init == PacketSizeInitEnum.PLE:
                        if size == 1500:
                            inital_embedding = torch.zeros(pe_size_embedding)
                            inital_embedding[:len(ple_size_bins)] = 1
                        else:
                            inital_embedding = torch.zeros(pe_size_embedding)
                            bin_idx = size // pe_size_ple_bin_size
                            start_bin = ple_size_bins[bin_idx]
                            relative_pos = (size - start_bin) / pe_size_ple_bin_size
                            inital_embedding[bin_idx] = relative_pos
                            if bin_idx >= 1:
                                inital_embedding[:bin_idx] = 1
                    packet_size_nn_embedding.weight.data[i, :] = inital_embedding
        if pe_ipt_processing == ProcessIPT.EMBED:
            # PLE initialization of IPT embeddings
            if pe_ipt_embedding == 4:
                ipt_bins_sections = (
                    torch.cat((torch.arange(0, 105, step=5), torch.arange(110, 260, step=10))),
                    torch.cat((torch.arange(275, 1025, step=25), torch.arange(1050, 2050, step=50))),
                    torch.arange(2100, 5100, step=100),
                    torch.cat((torch.arange(5250, 7750, step=250), torch.arange(8000, 10500, step=500), torch.arange(11000, 32000, step=1000))),
                )
            elif pe_ipt_embedding == 6:
                ipt_bins_sections = (
                    torch.arange(0, 11),
                    torch.cat((torch.arange(15, 105, step=5), torch.arange(110, 260, step=10))),
                    torch.cat((torch.arange(275, 1025, step=25), torch.arange(1050, 2050, step=50))),
                    torch.arange(2100, 5100, step=100),
                    torch.cat((torch.arange(5250, 7750, step=250), torch.arange(8000, 10500, step=500))),
                    torch.arange(11000, 32000, step=1000),
                )
            elif pe_ipt_embedding == 8:
                ipt_bins_sections = (
                    torch.tensor([0]),
                    torch.arange(1, 11, step=1),
                    torch.cat((torch.arange(15, 105, step=5), torch.arange(110, 260, step=10))),
                    torch.arange(275, 1025, step=25),
                    torch.arange(1050, 2050, step=50),
                    torch.arange(2100, 5100, step=100),
                    torch.cat((torch.arange(5250, 7750, step=250), torch.arange(8000, 15500, step=500))),
                    torch.arange(16000, 32000, step=1000),
                )
            elif pe_ipt_embedding == 10:
                ipt_bins_sections = (
                    torch.tensor([0]),
                    torch.arange(1, 11, step=1),
                    torch.cat((torch.arange(12, 32, step=2),  torch.arange(33, 63, step=3))),
                    torch.arange(65, 205, step=5),
                    torch.arange(210, 510, step=10),
                    torch.arange(525, 1025, step=25),
                    torch.arange(1050, 2050, step=50),
                    torch.arange(2100, 5100, step=100),
                    torch.cat((torch.arange(5250, 7750, step=250), torch.arange(8000, 15500, step=500))),
                    torch.arange(16000, 32000, step=1000),
                )
            elif pe_ipt_embedding == 12:
                ipt_bins_sections = (
                    torch.tensor([0]),
                    torch.arange(1, 11, step=1),
                    torch.arange(12, 32, step=2),
                    torch.arange(33, 63, step=3),
                    torch.arange(65, 205, step=5),
                    torch.arange(210, 510, step=10),
                    torch.arange(525, 1025, step=25),
                    torch.arange(1050, 2050, step=50),
                    torch.arange(2100, 5100, step=100),
                    torch.arange(5250, 7750, step=250),
                    torch.arange(8000, 15500, step=500),
                    torch.arange(16000, 32000, step=1000),
                )
            ipt_bins = torch.cat(ipt_bins_sections)
            packet_ipt_nn_embedding = nn.Embedding(num_embeddings=len(ipt_bins), embedding_dim=pe_ipt_embedding, padding_idx=0)
            i = 0
            for s in range(len(ipt_bins_sections)):
                segment_length = len(ipt_bins_sections[s])
                for b in range(segment_length):
                    inital_embedding = torch.zeros(pe_ipt_embedding)
                    if i != 0:
                        last_bin = ipt_bins_sections[s - 1][-1] if s > 0 else 0
                        inital_embedding[s] = ((ipt_bins_sections[s][b] - last_bin)  / (ipt_bins_sections[s][-1] - last_bin))
                        if s >= 1:
                            inital_embedding[:s] = 1
                    packet_ipt_nn_embedding.weight.data[i, :] = inital_embedding
                    i += 1
            ipt_bins[-1] = 2**32

        conv1_input_channels = pe_size_embedding + (pe_ipt_embedding if pe_ipt_processing == ProcessIPT.EMBED else 1 if pe_ipt_processing == ProcessIPT.DIRECT else 0)
        conv1_input_channels += 2 if pe_onehot_dirs else 1
        if stem_type == StemType.EMBED_CONV:
            stem = [
                conv(conv1_input_channels, out_channels, kernel_size=kernel_size, stride=1),
                norm(out_channels),
                nn.ReLU(inplace=True),
            ]
            stem_output_channels = out_channels
        else:
            stem_output_channels = conv1_input_channels
    return nn.Sequential(*stem) if len(stem) > 0 else nn.Identity(), stem_output_channels, packet_size_nn_embedding, packet_ipt_nn_embedding, ipt_bins

class Multimodal_CESNET_Enhanced(nn.Module):
    def __init__(self, num_classes: int = 0, flowstats_input_size: int = 0, ppi_input_channels: int = 3,
                       init_weights: bool = True, cnn_ppi_stem_type: StemType = StemType.EMBED,
                       pe_size_embedding: int = 20, pe_size_include_dir: bool = False, pe_size_init: PacketSizeInitEnum = PacketSizeInitEnum.PLE, pe_size_ple_bin_size: int = 100,
                       pe_ipt_processing: ProcessIPT = ProcessIPT.EMBED, pe_ipt_embedding: int = 10, pe_onehot_dirs: bool = True,
                       conv_normalization: NormalizationEnum = NormalizationEnum.BATCH_NORM, linear_normalization: NormalizationEnum = NormalizationEnum.BATCH_NORM, group_norm_groups: int = 16,
                       cnn_ppi_channels: tuple[int, ...] = (192, 256, 384, 448), cnn_ppi_strides: tuple[int, ...] = (1, 1, 1, 1), cnn_ppi_kernel_sizes: tuple[int, ...] = (7, 7, 5, 3),
                       cnn_ppi_use_stdconv: bool = False, cnn_ppi_downsample_avg: bool = True, cnn_ppi_blocks_dropout: float = 0.3, cnn_ppi_first_bottle_ratio: float = 0.25, cnn_ppi_dropout_type: DropoutType = DropoutType.REGULAR,
                       cnn_ppi_global_pool: GlobalPoolEnum = GlobalPoolEnum.GEM_3_LEARNABLE, cnn_ppi_global_pool_act: bool = False, cnn_ppi_global_pool_dropout: float = 0.0,
                       use_mlp_flowstats: bool = False, mlp_flowstats_size1: int = 256, mlp_flowstats_size2: int = 64, mlp_flowstats_num_hidden: int = 1, mlp_flowstats_dropout: float = 0.0,
                       use_mlp_shared: bool = True, mlp_shared_size: int = 448, mlp_shared_dropout: float = 0.0,
                       save_psizes_hist: bool = False,
                       ):
        super().__init__()
        cnn_ppi_stem_type = convert_str_to_enum(cnn_ppi_stem_type, enum_class=StemType)
        pe_size_init = convert_str_to_enum(pe_size_init, enum_class=PacketSizeInitEnum)
        pe_ipt_processing = convert_str_to_enum(pe_ipt_processing, enum_class=ProcessIPT)
        conv_normalization = convert_str_to_enum(conv_normalization, enum_class=NormalizationEnum)
        linear_normalization = convert_str_to_enum(linear_normalization, enum_class=NormalizationEnum)
        cnn_ppi_global_pool = convert_str_to_enum(cnn_ppi_global_pool, enum_class=GlobalPoolEnum)
        cnn_ppi_dropout_type = convert_str_to_enum(cnn_ppi_dropout_type, enum_class=DropoutType)
        if ppi_input_channels != 3:
            raise ValueError("ppi_input_channels must be 3 for now")
        if use_mlp_flowstats and flowstats_input_size == 0:
            raise ValueError("flowstats_input_size must be set when use_mlp_flowstats is used")
        if pe_size_init == PacketSizeInitEnum.WITH_DIR and not pe_size_include_dir:
            raise ValueError("packet_embedding_init cannot be with-dir when pe_size_include_dir is false")
        if pe_size_init == PacketSizeInitEnum.PLE and pe_size_include_dir:
            raise ValueError("packet_embedding_init cannot be PLE when pe_size_include_dir is true")
        if pe_size_init == PacketSizeInitEnum.PLE and (1500 // pe_size_ple_bin_size) > pe_size_embedding:
            raise ValueError("pe_size_embedding must be greater than the number of bins for PLE")
        if pe_ipt_processing == ProcessIPT.EMBED and pe_ipt_embedding not in (4, 6, 8, 10, 12):
            raise ValueError("pe_ipt_embedding must be 4, 6, 8, 10, or 12")

        self.num_classes = num_classes
        self.use_mlp_flowstats = use_mlp_flowstats
        self.use_mlp_shared = use_mlp_shared
        mlp_shared_input_size = cnn_ppi_channels[-1] + mlp_flowstats_size2 if use_mlp_flowstats else cnn_ppi_channels[-1]
        self.num_features = mlp_shared_size if self.use_mlp_shared else mlp_shared_input_size
        self.pe_size_include_dir = pe_size_include_dir
        self.pe_onehot_dirs = pe_onehot_dirs
        self.pe_ipt_processing = pe_ipt_processing
        conv = StdConv1d if cnn_ppi_use_stdconv else PadConv1d
        conv_norm = partial(conv_norm_from_enum, norm_enum=conv_normalization, group_norm_groups=group_norm_groups)
        linear_norm = partial(linear_norm_from_enum, norm_enum=linear_normalization)
        self.cnn_ppi_stem_type = cnn_ppi_stem_type
        stem = build_cnn_ppi_stem(stem_type=cnn_ppi_stem_type,
                                  ppi_input_channels=ppi_input_channels,
                                  out_channels=cnn_ppi_channels[0] // 2,
                                  kernel_size=7,
                                  conv=conv, norm=conv_norm,
                                  pe_size_embedding=pe_size_embedding,
                                  pe_size_include_dir=pe_size_include_dir,
                                  pe_size_init=pe_size_init,
                                  pe_size_ple_bin_size=pe_size_ple_bin_size,
                                  pe_ipt_embedding=pe_ipt_embedding,
                                  pe_ipt_processing=pe_ipt_processing,
                                  pe_onehot_dirs=pe_onehot_dirs,)
        self.cnn_ppi_stem, stem_output_channels, self.packet_size_nn_embedding, self.packet_ipt_nn_embedding, ipt_bins = stem
        if ipt_bins is not None:
            self.register_buffer("ipt_bins", ipt_bins)

        self.cnn_ppi = build_cnn_ppi(channels=cnn_ppi_channels,
                                     strides=cnn_ppi_strides,
                                     kernel_sizes=cnn_ppi_kernel_sizes,
                                     stem_output_channels=stem_output_channels,
                                     dropout_rate_path=cnn_ppi_blocks_dropout,
                                     dropout_type=cnn_ppi_dropout_type,
                                     first_block_bottle_ratio=cnn_ppi_first_bottle_ratio,
                                     downsample_avg=cnn_ppi_downsample_avg,
                                     conv=conv, norm=conv_norm)
        if cnn_ppi_global_pool == GlobalPoolEnum.AVG:
            gp = nn.AdaptiveAvgPool1d(output_size=1)
        elif cnn_ppi_global_pool == GlobalPoolEnum.MAX:
            gp = nn.AdaptiveMaxPool1d(output_size=1)
        elif cnn_ppi_global_pool == GlobalPoolEnum.GEM_3:
            gp = AdaptiveGeM(output_size=1, p=3.0)
        elif cnn_ppi_global_pool == GlobalPoolEnum.GEM_6:
            gp = AdaptiveGeM(output_size=1, p=6.0)
        elif cnn_ppi_global_pool == GlobalPoolEnum.GEM_3_LEARNABLE:
            gp = AdaptiveGeM(output_size=1, p=3.0, learnable_p=True)
        elif cnn_ppi_global_pool == GlobalPoolEnum.GEM_6_LEARNABLE:
            gp = AdaptiveGeM(output_size=1, p=6.0, learnable_p=True)
        self.cnn_ppi_global_pool = nn.Sequential(
            gp,
            nn.Flatten(),
            nn.Dropout(cnn_ppi_global_pool_dropout) if cnn_ppi_global_pool_dropout > 0 else nn.Identity(),
            nn.ReLU(inplace=True) if cnn_ppi_global_pool_act else nn.Identity(),
        )
        if self.use_mlp_flowstats:
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
                nn.Dropout(mlp_flowstats_dropout) if mlp_flowstats_dropout > 0 else nn.Identity(),
                nn.ReLU(inplace=True),
            )
        if self.use_mlp_shared:
            self.mlp_shared = nn.Sequential(
                nn.Linear(mlp_shared_input_size, mlp_shared_size),
                linear_norm(mlp_shared_size),
                nn.Dropout(mlp_shared_dropout) if mlp_shared_dropout > 0 else nn.Identity(),
                nn.ReLU(inplace=True),
            )
        if self.num_classes > 0:
            self.classifier = nn.Linear(self.num_features, num_classes)
        else:
            self.classifier = nn.Identity()
        if init_weights:
            self.apply(init_weights_fn)
        self.save_psizes_hist = save_psizes_hist
        if self.save_psizes_hist:
            if self.pe_size_include_dir:
                raise ValueError("save_psizes_hist cannot be used with pe_size_include_dir")
            self.register_buffer("psizes_hist", torch.zeros(1501, dtype=torch.int64))

    def forward_features(self, ppi, flowstats):
        if self.cnn_ppi_stem_type == StemType.EMBED or self.cnn_ppi_stem_type == StemType.EMBED_CONV:
            assert self.packet_size_nn_embedding is not None
            if self.pe_size_include_dir:
                size_embedding_input = (ppi[:, PPI_SIZE_POS, :] * ppi[:, PPI_DIR_POS, :]).int() + 1500
            else:
                size_embedding_input = ppi[:, PPI_SIZE_POS, :].int()
                if self.training and self.save_psizes_hist:
                    self.psizes_hist += torch.histc(size_embedding_input, bins=1501, min=0, max=1500)
            size = (self.packet_size_nn_embedding(size_embedding_input),)
            if self.pe_ipt_processing == ProcessIPT.DIRECT:
                ipt = (ppi[:, PPI_IPT_POS, :].unsqueeze(-1),)
            elif self.pe_ipt_processing == ProcessIPT.EMBED:
                assert self.packet_ipt_nn_embedding is not None
                ipt_embedding_input = torch.bucketize(ppi[:, PPI_IPT_POS, :].contiguous(), self.ipt_bins)
                ipt = (self.packet_ipt_nn_embedding(ipt_embedding_input),)
            else:
                ipt = ()
            if self.pe_onehot_dirs:
                dir = (
                    (ppi[:, PPI_DIR_POS, :] > 0).int().unsqueeze(-1),
                    (ppi[:, PPI_DIR_POS, :] < 0).int().unsqueeze(-1),)
            else:
                dir = (ppi[:, PPI_DIR_POS, :].unsqueeze(-1),)
            ppi_embedded = torch.cat(ipt + dir + size, dim=2).transpose(1, 2)
        else:
            ppi_embedded = ppi
        out = self.cnn_ppi_stem(ppi_embedded)
        out = self.cnn_ppi(out)
        out = self.cnn_ppi_global_pool(out)
        if self.use_mlp_flowstats:
            out_flowstats = self.mlp_flowstats(flowstats)
            out = torch.column_stack([out, out_flowstats])
        if self.use_mlp_shared:
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
