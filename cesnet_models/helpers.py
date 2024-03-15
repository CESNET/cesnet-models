from dataclasses import dataclass
from enum import Enum
from typing import Optional

from torch import nn
from torch.hub import load_state_dict_from_url


@dataclass
class Weights:
    """
    This class contains important attributes associated with the pre-trained weights.

    Args:
        url:
        datazoo_config:
        meta:
    """

    bucket_url: str
    file_name: str
    transforms: dict
    meta: dict

class WeightsEnum(Enum):
    """
    This class is used to group different pre-trained weights.
    """

    def get_state_dict(self, model_dir: Optional[str]) -> dict:
        url = f"{self.value.bucket_url}&file={self.value.file_name}"
        return load_state_dict_from_url(url=url, model_dir=model_dir, file_name=self.value.file_name, map_location="cpu", progress=True)
    
    @property
    def transforms(self):
        return self.value.transforms

    @property
    def meta(self):
        return self.value.meta

def count_nn_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
