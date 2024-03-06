from typing import Optional

import numpy as np

from cesnet_models.architectures.multimodal_cesnet import Multimodal_CESNET, NormalizationEnum
from cesnet_models.helpers import Weights, WeightsEnum
from cesnet_models.transforms import ClipAndScaleFlowstats, ClipAndScalePPI, NormalizeHistograms


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
            raise ValueError(f"Based on pretrained weights, num_classes should be {weights.value.meta['num_classes']} but got {num_classes}")
        if flowstats_input_size is not None and flowstats_input_size != weights.value.meta["flowstats_input_size"]:
            raise ValueError(f"Based on pretrained weights, flowstats_input_size should be {weights.value.meta['flowstats_input_size']} but got {flowstats_input_size}")
        if ppi_input_channels is not None and ppi_input_channels != weights.value.meta["ppi_input_channels"]:
            raise ValueError(f"Based on pretrained weights, ppi_input_channels should be {weights.value.meta['ppi_input_channels']} but got {ppi_input_channels}")
        num_classes = weights.value.meta["num_classes"]
        flowstats_input_size = weights.value.meta["flowstats_input_size"]
        ppi_input_channels = weights.value.meta["ppi_input_channels"]
    assert num_classes is not None and flowstats_input_size is not None and ppi_input_channels is not None

    model = Multimodal_CESNET(**model_configuration, num_classes=num_classes, flowstats_input_size=flowstats_input_size, ppi_input_channels=ppi_input_channels)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(model_dir=model_dir))
    return model

class MM_CESNET_QUIC1_Weights(WeightsEnum):
    CESNET_QUIC22_Week1 = Weights(
        url="www.github.com/foo.pth",
        training_dataset="CESNET_QUIC22",
        transforms_config={
            "ppi_transform": ClipAndScalePPI(
                psizes_scaler_enum="standard",
                psizes_scaler_attrs={"mean_": [628.7188066587469], "scale_": [554.9873890137749]},
                pszies_min=1,
                psizes_max=1500,
                ipt_scaler_enum="standard",
                ipt_scaler_attrs={"mean_": [693.3538472283111], "scale_": [5108.986448664004]},
                ipt_min=0,
                ipt_max=65000,),
            "flowstats_transform": ClipAndScaleFlowstats(
                flowstats_scaler_enum="robust",
                flowstats_scaler_attrs={
                    "center_": [2294.0, 5988.0, 13.0, 12.0, 0.9607470035552979, 11.0, 3.0, 0.23399999737739563],
                    "scale_": [2806.0, 4366.25, 9.0, 11.0, 17.460650961846113, 11.0, 2.0, 3.1197499781847]
                },
                flowstats_quantiles=[168878.8099999996, 4059212.0799999908, 1119.0, 3240.0099999999948, np.inf, np.inf, np.inf, np.inf,],
                quantile_clip=0.99,
            ),
            "flowstats_phist_transform": NormalizeHistograms(),
        },
        meta={
            "num_classes": 100,
            "classes": ["foo", "bar"],
            "ppi_input_channels": 3,
            "flowstats_input_size": 30,
            "use_packet_histograms": True,
            "num_params": 1_000_000,
            "paper_doi": "https://doi.org/10.23919/TMA58422.2023.10199052",
            "description":  """These weights reproduce the results of the paper."""
        }
    )
    DEFAULT = CESNET_QUIC22_Week1

def mm_cesnet_v2(weights: Optional[MM_CESNET_QUIC1_Weights] = None,
                      model_dir: Optional[str] = None,
                      num_classes: Optional[int] = None,
                      flowstats_input_size: Optional[int] = None,
                      ppi_input_channels: Optional[int] = None,
                      ) -> Multimodal_CESNET:
    """
    This is a second version of the multimodal CESNET architecture. It was used in
    the "Encrypted traffic classification: the QUIC case" paper.

    Changes from the first version:
        - Global pooling was added to the CNN part processing PPI sequences, instead of a simple flattening.
        - One more Conv1d layer was added to the CNN part and the number of channels was increased.
        - The size of the MLP processing flow statistics was increased.
        - The size of the MLP processing shared representations was decreased.
        - Some dropout rates were decreased.
    """
    v2_model_configuration = {
        "conv_normalization": NormalizationEnum.BATCH_NORM,
        "linear_normalization": NormalizationEnum.BATCH_NORM,
        "cnn_num_hidden": 3,
        "cnn_channels1": 200,
        "cnn_channels2": 300,
        "cnn_channels3": 300,
        "cnn_use_pooling": True,
        "cnn_dropout_rate": 0.1,
        "flowstats_num_hidden": 2,
        "flowstats_size": 225,
        "flowstats_out_size": 225,
        "flowstats_dropout_rate": 0.1,
        "latent_num_hidden":  0,
        "latent_size": 600,
        "latent_dropout_rate": 0.2,
    }
    return _multimodal_cesnet(model_configuration=v2_model_configuration,
                              weights=weights,
                              model_dir=model_dir,
                              num_classes=num_classes,
                              flowstats_input_size=flowstats_input_size,
                              ppi_input_channels=ppi_input_channels)

def mm_cesnet_v1(weights: Optional[MM_CESNET_QUIC1_Weights] = None,
                      model_dir: Optional[str] = None,
                      num_classes: Optional[int] = None,
                      flowstats_input_size: Optional[int] = None,
                      ppi_input_channels: Optional[int] = None,
                      ) -> Multimodal_CESNET:
    """
    This model was used in the "Fine-grained TLS services classification with reject option" paper.
    """
    v1_model_configuration = {
        "conv_normalization": NormalizationEnum.BATCH_NORM,
        "linear_normalization": NormalizationEnum.BATCH_NORM,
        "cnn_num_hidden": 2,
        "cnn_channels1": 72,
        "cnn_channels2": 128,
        "cnn_channels3": 128,
        "cnn_use_pooling": False,
        "cnn_dropout_rate": 0.2,
        "flowstats_num_hidden": 2,
        "flowstats_size": 64,
        "flowstats_out_size": 32,
        "flowstats_dropout_rate": 0.2,
        "latent_num_hidden": 1,
        "latent_size": 480,
        "latent_dropout_rate": 0.2,
    }
    return _multimodal_cesnet(model_configuration=v1_model_configuration,
                              weights=weights,
                              model_dir=model_dir,
                              num_classes=num_classes,
                              flowstats_input_size=flowstats_input_size,
                              ppi_input_channels=ppi_input_channels)
