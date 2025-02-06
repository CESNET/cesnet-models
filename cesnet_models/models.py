from typing import Optional

import numpy as np

from cesnet_models._models_meta import (_CESNET_QUIC22_102_CLASSES,
                                        _CESNET_QUIC22_WEEK46_DOMAIN_SPLIT,
                                        _CESNET_TLS22_191_CLASSES)
from cesnet_models.architectures.embedding_model_wrapper import EmbeddingModel
from cesnet_models.architectures.multimodal_cesnet import Multimodal_CESNET, NormalizationEnum
from cesnet_models.architectures.multimodal_cesnet_enhanced import (GlobalPoolEnum,
                                                                    Multimodal_CESNET_Enhanced,
                                                                    PacketSizeInitEnum, ProcessIPT,
                                                                    StemType)
from cesnet_models.helpers import Weights, WeightsEnum
from cesnet_models.transforms import ClipAndScaleFlowstats, ClipAndScalePPI, NormalizeHistograms


class Model_30pktTCNET_256_Weights(WeightsEnum):
    CESNET_QUIC22_Week46_Domains = Weights(
        bucket_url="https://liberouter.org/datazoo/download?bucket=cesnet-models",
        file_name="30pktTCNET_256_CESNET_QUIC22_Week46_Domains.pth",
        transforms={
            "ppi_transform": ClipAndScalePPI(
                psizes_scaler_enum="no-scaling",
                psizes_scaler_attrs=None,
                psizes_min=1,
                psizes_max=1500,
                ipt_scaler_enum="no-scaling",
                ipt_scaler_attrs=None,
                ipt_min=0,
                ipt_max=65000,),
            "flowstats_transform": None,
            "flowstats_phist_transform": None,
        },
        meta={
            "train_dataset": "CESNET-QUIC22",
            "train_dataset_size": "ORIG",
            "train_period_name": "W-2022-46",
            "domains": _CESNET_QUIC22_WEEK46_DOMAIN_SPLIT,
            "ppi_input_channels": 3,
            "num_params": 1_004_117,
            "paper_doi": "",
            "description":  "",
        }
    )
    DEFAULT = CESNET_QUIC22_Week46_Domains

def model_30pktTCNET_256(weights: Optional[Model_30pktTCNET_256_Weights] = None,
                         model_dir: Optional[str] = None) -> EmbeddingModel:
    """
    A single-modal neural network processing sequences of 30 packets and outputting 256-dimensional flow embeddings.
    For fine-tuning, consider using just the `backbone_model` attribute (an instance of Multimodal_CESNET_Enhanced) of the returned model.

    Parameters:
        weights: If provided, the model will be initialized with these weights.
        model_dir: If weights are provided, this folder will be used to store the weights.
    """
    architecture_params = {
        "use_mlp_flowstats": False,
        "init_weights": True,
        "cnn_ppi_stem_type": StemType.EMBED,
        "pe_size_embedding": 20,
        "pe_size_include_dir": False,
        "pe_size_init": PacketSizeInitEnum.PLE,
        "pe_size_ple_bin_size": 100,
        "pe_ipt_processing": ProcessIPT.EMBED,
        "pe_ipt_embedding": 10,
        "pe_onehot_dirs": True,
        "conv_normalization": NormalizationEnum.BATCH_NORM,
        "linear_normalization": NormalizationEnum.BATCH_NORM,
        "cnn_ppi_channels": [192, 256, 384, 448],
        "cnn_ppi_strides": [1, 1, 1, 1],
        "cnn_ppi_kernel_sizes": [7, 7, 5, 3],
        "cnn_ppi_use_stdconv": False,
        "cnn_ppi_downsample_avg": True,
        "cnn_ppi_blocks_dropout": 0.3,
        "cnn_ppi_first_bottle_ratio": 0.25,
        "cnn_ppi_global_pool": GlobalPoolEnum.GEM_3_LEARNABLE,
        "cnn_ppi_global_pool_act": False,
        "cnn_ppi_global_pool_dropout": 0.0,
        "use_mlp_shared": True,
        "mlp_shared_size": 448,
        "mlp_shared_dropout": 0.0
    }
    embedding_size = 256

    backbone_model = Multimodal_CESNET_Enhanced(**architecture_params, save_psizes_hist=True)
    model = EmbeddingModel(backbone_model, embedding_size=embedding_size)
    if weights is not None:
        state_dict = weights.get_state_dict(model_dir=model_dir)
        state_dict.pop("arcface_module.W", None)
        model.load_state_dict(state_dict)
        model.eval()
    return model

class MM_CESNET_V2_Weights(WeightsEnum):
    CESNET_QUIC22_Week44 = Weights(
        bucket_url="https://liberouter.org/datazoo/download?bucket=cesnet-models",
        file_name="mmv2_CESNET_QUIC22_Week44.pth",
        transforms={
            "ppi_transform": ClipAndScalePPI(
                psizes_scaler_enum="standard",
                psizes_scaler_attrs={"mean_": [473.2243172415956], "scale_": [529.8190065488045]},
                psizes_min=1,
                psizes_max=1460,
                ipt_scaler_enum="standard",
                ipt_scaler_attrs={"mean_": [105.15864160400703], "scale_": [1061.1513005552956]},
                ipt_min=0,
                ipt_max=15000,),
            "flowstats_transform": ClipAndScaleFlowstats(
                flowstats_scaler_enum="robust",
                flowstats_scaler_attrs={
                    "center_": [4176.5, 5267.0, 12.0, 13.0, 0.2200635001063347, 25.0, 4.0, 0.15600000321865082],
                    "scale_": [4422.0, 7358.0, 14.0, 15.0, 4.944558419287205, 13.0, 3.0, 0.5030000135302544]},
                flowstats_quantiles=[134147.9500000002, 3182444.600000009, 535.0, 2667.0, np.inf, np.inf, np.inf, np.inf],
                quantile_clip=0.99,),
            "flowstats_phist_transform": NormalizeHistograms(),
        },
        meta={
            "train_dataset": "CESNET-QUIC22",
            "train_dataset_size": "ORIG",
            "train_period_name": "W-2022-44",
            "num_classes": 102,
            "classes": _CESNET_QUIC22_102_CLASSES,
            "use_tcp_features": False,
            "use_packet_histograms": True,
            "ppi_input_channels": 3,
            "flowstats_input_size": 43,
            "flowstats_features": [
                "BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV", "DURATION",
                "PPI_LEN", "PPI_ROUNDTRIPS", "PPI_DURATION",
                "FLOW_ENDREASON_IDLE", "FLOW_ENDREASON_ACTIVE", "FLOW_ENDREASON_OTHER",
                "PSIZE_BIN1", "PSIZE_BIN2", "PSIZE_BIN3", "PSIZE_BIN4", "PSIZE_BIN5", "PSIZE_BIN6","PSIZE_BIN7", "PSIZE_BIN8",
                "PSIZE_BIN1_REV", "PSIZE_BIN2_REV", "PSIZE_BIN3_REV", "PSIZE_BIN4_REV", "PSIZE_BIN5_REV", "PSIZE_BIN6_REV", "PSIZE_BIN7_REV", "PSIZE_BIN8_REV",
                "IPT_BIN1", "IPT_BIN2", "IPT_BIN3", "IPT_BIN4", "IPT_BIN5", "IPT_BIN6", "IPT_BIN7", "IPT_BIN8",
                "IPT_BIN1_REV", "IPT_BIN2_REV", "IPT_BIN3_REV", "IPT_BIN4_REV", "IPT_BIN5_REV", "IPT_BIN6_REV", "IPT_BIN7_REV", "IPT_BIN8_REV"],
            "num_params": 2_261_653,
            "paper_doi": "https://doi.org/10.23919/TMA58422.2023.10199052",
            "description": """These weights reproduce the results of the "Encrypted traffic classification: the QUIC case" paper."""
        }
    )
    DEFAULT = CESNET_QUIC22_Week44

def _multimodal_cesnet(model_configuration: dict,
                       weights: Optional[WeightsEnum],
                       num_classes: Optional[int],
                       flowstats_input_size: Optional[int],
                       ppi_input_channels: Optional[int],
                       model_dir: Optional[str],) -> Multimodal_CESNET:
    if weights is None:
        if num_classes is None:
            raise ValueError("num_classes must be provided when weights are not provided")
        if flowstats_input_size is None:
            raise ValueError("flowstats_input_size must be provided when weights are not provided")
        if ppi_input_channels is None:
            raise ValueError("ppi_input_channels must be provided when weights are not provided")
    if weights is not None:
        if num_classes is not None and num_classes != weights.value.meta["num_classes"]:
            raise ValueError(f"Based on pre-trained weights, num_classes should be {weights.value.meta['num_classes']} but got {num_classes}")
        if flowstats_input_size is not None and flowstats_input_size != weights.value.meta["flowstats_input_size"]:
            raise ValueError(f"Based on pre-trained weights, flowstats_input_size should be {weights.value.meta['flowstats_input_size']} but got {flowstats_input_size}")
        if ppi_input_channels is not None and ppi_input_channels != weights.value.meta["ppi_input_channels"]:
            raise ValueError(f"Based on pre-trained weights, ppi_input_channels should be {weights.value.meta['ppi_input_channels']} but got {ppi_input_channels}")
        num_classes = weights.value.meta["num_classes"]
        flowstats_input_size = weights.value.meta["flowstats_input_size"]
        ppi_input_channels = weights.value.meta["ppi_input_channels"]
    assert num_classes is not None and flowstats_input_size is not None and ppi_input_channels is not None

    model = Multimodal_CESNET(**model_configuration, num_classes=num_classes, flowstats_input_size=flowstats_input_size, ppi_input_channels=ppi_input_channels)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(model_dir=model_dir))
        model.eval()
    return model

def mm_cesnet_v2(weights: Optional[MM_CESNET_V2_Weights] = None,
                 model_dir: Optional[str] = None,
                 num_classes: Optional[int] = None,
                 flowstats_input_size: Optional[int] = None,
                 ppi_input_channels: Optional[int] = None,
                 ) -> Multimodal_CESNET:
    """
    This is a second version of the multimodal CESNET architecture. It was used in
    the *"Encrypted traffic classification: the QUIC case"* paper.

    Changes from the first version:
        - Global pooling was added to the CNN part processing PPI sequences, instead of a simple flattening.
        - One more Conv1D layer was added to the CNN part and the number of channels was increased.
        - The size of the MLP processing flow statistics was increased.
        - The size of the MLP processing shared representations was decreased.
        - Some dropout rates were decreased.

    Parameters:
        weights: If provided, the model will be initialized with these weights.
        model_dir: If weights are provided, this folder will be used to store the weights.
        num_classes: Number of classes.
        flowstats_input_size: Size of the flow statistics input.
        ppi_input_channels: Number of channels in the PPI input.
    """
    v2_model_configuration = {
        "conv_normalization": NormalizationEnum.BATCH_NORM,
        "linear_normalization": NormalizationEnum.BATCH_NORM,
        "cnn_ppi_num_blocks": 3,
        "cnn_ppi_channels1": 200,
        "cnn_ppi_channels2": 300,
        "cnn_ppi_channels3": 300,
        "cnn_ppi_use_pooling": True,
        "cnn_ppi_dropout_rate": 0.1,
        "mlp_flowstats_num_hidden": 2,
        "mlp_flowstats_size1": 225,
        "mlp_flowstats_size2": 225,
        "mlp_flowstats_dropout_rate": 0.1,
        "mlp_shared_num_hidden":  0,
        "mlp_shared_size": 600,
        "mlp_shared_dropout_rate": 0.2,
    }
    return _multimodal_cesnet(model_configuration=v2_model_configuration,
                              weights=weights,
                              model_dir=model_dir,
                              num_classes=num_classes,
                              flowstats_input_size=flowstats_input_size,
                              ppi_input_channels=ppi_input_channels)

class MM_CESNET_V1_Weights(WeightsEnum):
    CESNET_TLS22_WEEK40 = Weights(
        bucket_url="https://liberouter.org/datazoo/download?bucket=cesnet-models",
        file_name="mmv1_CESNET_TLS22_Week40.pth",
        transforms={
            "ppi_transform": ClipAndScalePPI(
                psizes_scaler_enum="standard",
                psizes_scaler_attrs={"mean_": [708.3937210483823],"scale_": [581.2441777831351]},
                psizes_min=1,
                psizes_max=1460,
                ipt_scaler_enum="standard",
                ipt_scaler_attrs={"mean_": [228.10927542399668],"scale_": [1517.1576685053515]},
                ipt_min=1,
                ipt_max=15000,),
            "flowstats_transform": ClipAndScaleFlowstats(
                flowstats_scaler_enum="robust",
                flowstats_scaler_attrs={
                    "center_": [2493.0, 6358.0, 13.0, 13.0, 0.5710994899272919, 13.0, 3.0, 0.20399999618530273],
                    "scale_": [3025.0, 5321.0, 9.0, 10.0, 5.814603120088577, 11.0, 2.0, 0.7700000107288361]},
                flowstats_quantiles=[19147.0, 125204.75, 78.0, 126.0, np.inf, np.inf, np.inf, np.inf],
                quantile_clip=0.95,),
            "flowstats_phist_transform": None,
        },
        meta={
            "train_dataset": "CESNET-TLS22",
            "train_dataset_size": "ORIG",
            "train_period_name": "W-2021-40",
            "num_classes": 191,
            "classes": _CESNET_TLS22_191_CLASSES,
            "use_tcp_features": True,
            "use_packet_histograms": False,
            "ppi_input_channels": 3,
            "flowstats_input_size": 17,
            "flowstats_features": [
                "BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV", "DURATION",
                "PPI_LEN", "PPI_ROUNDTRIPS", "PPI_DURATION",
                "FLAG_CWR", "FLAG_CWR_REV", "FLAG_ECE", "FLAG_ECE_REV", "FLAG_PSH_REV", "FLAG_RST", "FLAG_RST_REV", "FLAG_FIN", "FLAG_FIN_REV"],
            "num_params": 1_217_903,
            "paper_doi": "https://doi.org/10.1016/j.comnet.2022.109467",
            "description":  """These weights reproduce the results of the "Fine-grained TLS services classification with reject option" paper."""
        }
    )
    DEFAULT = CESNET_TLS22_WEEK40

def mm_cesnet_v1(weights: Optional[MM_CESNET_V1_Weights] = None,
                 model_dir: Optional[str] = None,
                 num_classes: Optional[int] = None,
                 flowstats_input_size: Optional[int] = None,
                 ppi_input_channels: Optional[int] = None,
                 ) -> Multimodal_CESNET:
    """
    This model was used in the *"Fine-grained TLS services classification with reject option"* paper.

    Parameters:
        weights: If provided, the model will be initialized with these weights.
        model_dir: If weights are provided, this folder will be used to store the weights.
        num_classes: Number of classes.
        flowstats_input_size: Size of the flow statistics input.
        ppi_input_channels: Number of channels in the PPI input.
    """
    v1_model_configuration = {
        "conv_normalization": NormalizationEnum.BATCH_NORM,
        "linear_normalization": NormalizationEnum.BATCH_NORM,
        "cnn_ppi_num_blocks": 2,
        "cnn_ppi_channels1": 72,
        "cnn_ppi_channels2": 128,
        "cnn_ppi_channels3": 128,
        "cnn_ppi_use_pooling": False,
        "cnn_ppi_dropout_rate": 0.2,
        "mlp_flowstats_num_hidden": 2,
        "mlp_flowstats_size1": 64,
        "mlp_flowstats_size2": 32,
        "mlp_flowstats_dropout_rate": 0.2,
        "mlp_shared_num_hidden": 1,
        "mlp_shared_size": 480,
        "mlp_shared_dropout_rate": 0.2,
    }
    return _multimodal_cesnet(model_configuration=v1_model_configuration,
                              weights=weights,
                              model_dir=model_dir,
                              num_classes=num_classes,
                              flowstats_input_size=flowstats_input_size,
                              ppi_input_channels=ppi_input_channels)
