from dataclasses import dataclass
from enum import Enum
from typing import Optional

from torch.hub import load_state_dict_from_url


@dataclass
class Weights:
    """
    This class is used to group important attributes associated with the pre-trained weights.

    Args:
        url:
        datazoo_config:
        meta:
    """

    url: str
    training_dataset: str
    transforms_config: dict
    meta: dict

class WeightsEnum(Enum):
    """
    This class is used to group the different pre-trained weights.
    """

    def get_state_dict(self, model_dir: Optional[str]) -> dict:
        return load_state_dict_from_url(url=self.value.url, model_dir=model_dir, progress=True, check_hash=True)
