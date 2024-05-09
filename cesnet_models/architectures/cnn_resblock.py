import torch
import torch.nn.functional as F
from torch import nn


class SimpleResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
            super().__init__()

            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(out_channels),
            )
            self.cnn_skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = self.cnn(x)
        out += self.cnn_skip(x)
        out = F.relu(out)
        return out

class CNN_ResBlock(nn.Module):
    """CNN with residual connections for processing packet sequences.
        - Used in "Data Augmentation for Traffic Classification"
        - https://arxiv.org/pdf/2401.10754
    """

    def __init__(self,
                 num_classes: int,
                 ppi_input_channels: int = 3,
                 channels1: int = 64,
                 channels2: int = 128,
                 ):
            super().__init__()

            self.conv1 = nn.Conv1d(ppi_input_channels, channels1, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm1d(channels1)
            self.block1 = SimpleResBlock(channels1, channels1)
            self.block2 = SimpleResBlock(channels1, channels2)
            self.classifier = nn.Linear(channels2, num_classes)

    def _forward_impl(self, ppi):
        out = self.conv1(ppi)
        out = self.bn1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = F.adaptive_avg_pool1d(out, 1)
        out = torch.flatten(out, 1)
        x = self.classifier(out)
        return x

    def forward(self, x):
        ppi, flowstats = x
        return self._forward_impl(ppi=ppi)
