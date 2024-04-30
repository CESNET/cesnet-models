from enum import Enum
from typing import Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch import nn
from typing_extensions import assert_never

from cesnet_models.constants import PHIST_BIN_COUNT, PPI_DIR_POS, PPI_IPT_POS, PPI_SIZE_POS


def get_scaler_attrs(scaler: StandardScaler | RobustScaler | MinMaxScaler) -> dict[str, list[float]]:
    if isinstance(scaler, StandardScaler):
        assert hasattr(scaler, "mean_") and scaler.mean_ is not None and hasattr(scaler, "scale_") and scaler.scale_ is not None
        scaler_attrs = {"mean_": scaler.mean_.tolist(), "scale_": scaler.scale_.tolist()}
    elif isinstance(scaler, RobustScaler):
        assert hasattr(scaler, "center_") and hasattr(scaler, "scale_")
        scaler_attrs = {"center_": scaler.center_.tolist(), "scale_": scaler.scale_.tolist()}
    elif isinstance(scaler, MinMaxScaler):
        assert hasattr(scaler, "min_") and hasattr(scaler, "scale_")
        scaler_attrs = {"min_": scaler.min_.tolist(), "scale_": scaler.scale_.tolist()}
    return scaler_attrs

def set_scaler_attrs(scaler: StandardScaler | RobustScaler | MinMaxScaler, scaler_attrs: dict[str, list[float]]):
    if isinstance(scaler, StandardScaler):
        assert "mean_" in scaler_attrs and "scale_" in scaler_attrs
        scaler.mean_ = np.array(scaler_attrs["mean_"])
        scaler.scale_ = np.array(scaler_attrs["scale_"])
    elif isinstance(scaler, RobustScaler):
        assert "center_" in scaler_attrs and "scale_" in scaler_attrs
        scaler.center_ = np.array(scaler_attrs["center_"])
        scaler.scale_ = np.array(scaler_attrs["scale_"])
    elif isinstance(scaler, MinMaxScaler):
        assert "min_" in scaler_attrs and "scale_" in scaler_attrs
        scaler.min_ = np.array(scaler_attrs["min_"])
        scaler.scale_ = np.array(scaler_attrs["scale_"])
    else:
        assert_never(scaler)

class ScalerEnum(str, Enum):
    """Available scalers for flow statistics, packet sizes, and inter-packet times."""
    STANDARD = "standard"
    """Standardize features by removing the mean and scaling to unit variance - [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)."""
    ROBUST = "robust"
    """Robust scaling with the median and the interquartile range - [`RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)."""
    MINMAX = "minmax"
    """Scaling to a (0, 1) range - [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)."""
    NO_SCALING = "no-scaling"
    def __str__(self): return self.value

class ClipAndScalePPI(nn.Module):
    """
    Transform class for scaling of per-packet information (PPI) sequences. This transform clips packet sizes and inter-packet times and scales them using a specified scaler.
    This class inherits from `nn.Module`, and the data transformation is implemented in the `forward` method.

    When used with the `cesnet-datazoo` package, the transform will be fitted during dataset initialization. Otherwise, the `psizes_scaler_attrs` and `ipt_scaler_attrs` must be provided.
    The required entries in `psizes_scaler_attrs` and `ipt_scaler_attrs` depend on the scaler used.

    - For `StandardScaler`, the required attributes are `mean_` and `scale_`.
    - For `RobustScaler`, the required attributes are `center_` and `scale_`.
    - For `MinMaxScaler`,  the required attributes `min_` and `scale_`.

    Expected format of input PPI sequences: `(batch_size, ppi_length, ppi_channels)`

    !!! info Padding
        The zero padding in PPI sequences is preserved during scaling, i.e., the padding zeroes are kept in the output.

    Parameters:
        psizes_scaler_enum: What scaler should be used for packet sizes. Options are standard, robust, minmax, and no-scaling.
        ipt_scaler_enum: What scaler should be used for inter-packet times. Options are standard, robust, minmax, and no-scaling.
        pszies_min: Clip packet sizes to this minimum value.
        psizes_max: Clip packet sizes to this maximum value.
        ipt_min: Clip inter-packet times to this minimum value.
        ipt_max: Clip inter-packet times to this maximum value.
        psizes_scaler_attrs: To use a pre-fitted packet sizes scaler, provide its attributes here.
        ipt_scaler_attrs: To use a pre-fitted inter-packet times scaler, provide its attributes here.
    """
    psizes_scaler: StandardScaler | RobustScaler | MinMaxScaler | None
    ipt_scaler: StandardScaler | RobustScaler | MinMaxScaler | None
    pszies_min: int
    psizes_max: int
    ipt_min: int
    ipt_max: int

    def __init__(self,
                 psizes_scaler_enum: ScalerEnum | str = ScalerEnum.STANDARD,
                 ipt_scaler_enum: ScalerEnum | str = ScalerEnum.STANDARD,
                 pszies_min: int = 1,
                 psizes_max: int = 1500,
                 ipt_min: int = 0,
                 ipt_max: int = 65000,
                 psizes_scaler_attrs: Optional[dict[str, list[float]]] = None,
                 ipt_scaler_attrs: Optional[dict[str, list[float]]] = None) -> None:
        super().__init__()
        if psizes_scaler_enum == ScalerEnum.STANDARD:
            self.psizes_scaler = StandardScaler()
        elif psizes_scaler_enum == ScalerEnum.ROBUST:
            self.psizes_scaler = RobustScaler()
        elif psizes_scaler_enum == ScalerEnum.MINMAX:
            self.psizes_scaler = MinMaxScaler()
        elif psizes_scaler_enum == ScalerEnum.NO_SCALING:
            self.psizes_scaler = None
        else:
            raise ValueError(f"psizes_scaler_enum must be one of {ScalerEnum.__members__}")
        if ipt_scaler_enum == ScalerEnum.STANDARD:
            self.ipt_scaler = StandardScaler()
        elif ipt_scaler_enum == ScalerEnum.ROBUST:
            self.ipt_scaler = RobustScaler()
        elif ipt_scaler_enum == ScalerEnum.MINMAX:
            self.ipt_scaler = MinMaxScaler()
        elif ipt_scaler_enum == ScalerEnum.NO_SCALING:
            self.ipt_scaler = None
        else:
            raise ValueError(f"ipt_scaler_enum must be one of {ScalerEnum.__members__}")
        self.pszies_min = pszies_min
        self.psizes_max = psizes_max
        self.ipt_max = ipt_max
        self.ipt_min = ipt_min
        self._psizes_scaler_enum = psizes_scaler_enum
        self._ipt_scaler_enum = ipt_scaler_enum

        if self.psizes_scaler and psizes_scaler_attrs is not None:
            set_scaler_attrs(scaler=self.psizes_scaler, scaler_attrs=psizes_scaler_attrs)
        if self.ipt_scaler and ipt_scaler_attrs is not None:
            set_scaler_attrs(scaler=self.ipt_scaler, scaler_attrs=ipt_scaler_attrs)
        self.needs_fitting = (self.ipt_scaler and ipt_scaler_attrs is None) or (self.psizes_scaler and psizes_scaler_attrs is None)

    def forward(self, x_ppi: np.ndarray) -> np.ndarray:
        if self.needs_fitting:
            raise ValueError("Scalers need to be fitted before using the ClipAndScalePPI transform")
        x_ppi = x_ppi.transpose(0, 2, 1)
        orig_shape = x_ppi.shape
        ppi_channels = x_ppi.shape[-1]
        x_ppi = x_ppi.reshape(-1, ppi_channels)
        x_ppi[:, PPI_IPT_POS] = x_ppi[:, PPI_IPT_POS].clip(max=self.ipt_max, min=self.ipt_min)
        x_ppi[:, PPI_SIZE_POS] = x_ppi[:, PPI_SIZE_POS].clip(max=self.psizes_max, min=self.pszies_min)
        padding_mask = x_ppi[:, PPI_DIR_POS] == 0 # Mask of zero padding
        if self.ipt_scaler is not None:
            x_ppi[:, PPI_IPT_POS] = self.ipt_scaler.transform(x_ppi[:, PPI_IPT_POS].reshape(-1, 1)).reshape(-1) # type: ignore
        if self.psizes_scaler is not None:
            x_ppi[:, PPI_SIZE_POS] = self.psizes_scaler.transform(x_ppi[:, PPI_SIZE_POS].reshape(-1, 1)).reshape(-1) # type: ignore
        x_ppi[padding_mask, PPI_IPT_POS] = 0
        x_ppi[padding_mask, PPI_SIZE_POS] = 0
        x_ppi = x_ppi.reshape(orig_shape).transpose(0, 2, 1)
        return x_ppi

    def to_dict(self) -> dict:
        d = {
            "psizes_scaler_enum": str(self._psizes_scaler_enum),
            "psizes_scaler_attrs": get_scaler_attrs(self.psizes_scaler) if self.psizes_scaler is not None else None,
            "pszies_min": self.pszies_min,
            "psizes_max": self.psizes_max,
            "ipt_scaler_enum": str(self._ipt_scaler_enum),
            "ipt_scaler_attrs": get_scaler_attrs(self.ipt_scaler) if self.ipt_scaler is not None else None,
            "ipt_min": self.ipt_min,
            "ipt_max": self.ipt_max,
        }
        return d

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(psizes_scaler={self._psizes_scaler_enum}, ipt_scaler={self._ipt_scaler_enum}, pszies_min={self.pszies_min}, psizes_max={self.psizes_max}, ipt_min={self.ipt_min}, ipt_max={self.ipt_max})"

class ClipAndScaleFlowstats(nn.Module):
    """
    Transform class for scaling of features describing an entire network flow -- called flow statistics. This transform clips flow statistics to their `quantile_clip` quantile and scales them using a specified scaler.
    This class inherits from `nn.Module`, and the data transformation is implemented in the `forward` method.

    When used with the `cesnet-datazoo` package, the transform will be fitted during dataset initialization. Otherwise, the `flowstats_scaler_attrs` must be provided.
    The required entries in `flowstats_scaler_attrs` depend on the scaler used.

    - For `StandardScaler`, the required attributes are `mean_` and `scale_`.
    - For `RobustScaler`, the required attributes are `center_` and `scale_`.
    - For `MinMaxScaler`,  the required attributes `min_` and `scale_`.

    Expected format of input flow statistics: `(batch_size, flowstats_features)`

    Parameters:
        flowstats_scaler_enum: What scaler should be used for flow statistics. Options are standard, robust, and minmax.
        quantile_clip: Clip flow statistics to this quantile.
        flowstats_quantiles:  To use pre-fitted quantiles, provide them here.
        flowstats_scaler_attrs: To use a pre-fitted scaler, provide its attributes here.
    """
    flowstats_scaler: StandardScaler | RobustScaler | MinMaxScaler
    quantile_clip: float
    flowstats_quantiles: Optional[list[float]]

    def __init__(self,
                 flowstats_scaler_enum: ScalerEnum | str = ScalerEnum.ROBUST,
                 quantile_clip: float = 0.99,
                 flowstats_quantiles: Optional[list[float]] = None,
                 flowstats_scaler_attrs: Optional[dict[str, list[float]]] = None) -> None:
        super().__init__()
        if flowstats_scaler_enum == ScalerEnum.STANDARD:
            self.flowstats_scaler = StandardScaler()
        elif flowstats_scaler_enum == ScalerEnum.ROBUST:
            self.flowstats_scaler = RobustScaler()
        elif flowstats_scaler_enum == ScalerEnum.MINMAX:
            self.flowstats_scaler = MinMaxScaler()
        else:
            raise ValueError(f"flowstats_scaler_enum must be one of {ScalerEnum.__members__}")
        self.quantile_clip = quantile_clip
        self._flowstats_scaler_enum = flowstats_scaler_enum

        if flowstats_scaler_attrs is None and flowstats_quantiles is None:
            self.needs_fitting = True
        elif flowstats_scaler_attrs is not None and flowstats_quantiles is not None:
            set_scaler_attrs(scaler=self.flowstats_scaler, scaler_attrs=flowstats_scaler_attrs)
            self.flowstats_quantiles = flowstats_quantiles
            self.needs_fitting = False
        else:
            raise ValueError("flowstats_quantiles and flowstats_scaler_attrs must be both set or both None")

    def forward(self, x_flowstats: np.ndarray) -> np.ndarray:
        if self.needs_fitting:
            raise ValueError("Scalers and quantiles need to be fitted before using this transform")
        x_flowstats = x_flowstats.clip(min=0, max=self.flowstats_quantiles)
        x_flowstats = self.flowstats_scaler.transform(x_flowstats) # type: ignore
        return x_flowstats
    
    def to_dict(self) -> dict:
        d = {
            "flowstats_scaler_enum": str(self._flowstats_scaler_enum),
            "flowstats_scaler_attrs": get_scaler_attrs(self.flowstats_scaler),
            "flowstats_quantiles": self.flowstats_quantiles,
            "quantile_clip": self.quantile_clip,
        }
        return d

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(flowstats_scaler={self._flowstats_scaler_enum}, quantile_clip={self.quantile_clip})"

class NormalizeHistograms(nn.Module):
    """
    Transform class for normalizing packet histograms.
    This class inherits from `nn.Module`, and the data transformation is implemented in the `forward` method.

    Expected format of input packet histograms: `(batch_size, 4 * PHIST_BIN_COUNT)`.
    The input histograms are expected to be in the following order: source packet sizes, destination packet sizes, source inter-packet times, and destination inter-packet times.
    Each of the four histograms is expected to have `PHIST_BIN_COUNT` bins, which is 8 in the current implementation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.bins = PHIST_BIN_COUNT

    def forward(self, x_flowstats_phist: np.ndarray) -> np.ndarray:
        src_sizes_pkt_count = x_flowstats_phist[:, :self.bins].sum(axis=1)[:, np.newaxis]
        dst_sizes_pkt_count = x_flowstats_phist[:, self.bins:(2*self.bins)].sum(axis=1)[:, np.newaxis]
        np.divide(x_flowstats_phist[:, :self.bins], src_sizes_pkt_count, out=x_flowstats_phist[:, :self.bins], where=src_sizes_pkt_count != 0)
        np.divide(x_flowstats_phist[:, self.bins:(2*self.bins)], dst_sizes_pkt_count, out=x_flowstats_phist[:, self.bins:(2*self.bins)], where=dst_sizes_pkt_count != 0)
        np.divide(x_flowstats_phist[:, (2*self.bins):(3*self.bins)], src_sizes_pkt_count - 1, out=x_flowstats_phist[:, (2*self.bins):(3*self.bins)], where=src_sizes_pkt_count > 1)
        np.divide(x_flowstats_phist[:, (3*self.bins):(4*self.bins)], dst_sizes_pkt_count - 1, out=x_flowstats_phist[:, (3*self.bins):(4*self.bins)], where=dst_sizes_pkt_count > 1)
        return x_flowstats_phist

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
