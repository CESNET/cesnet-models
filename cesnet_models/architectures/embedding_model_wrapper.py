from torch import nn
from torch.nn import functional as F


class EmbeddingModel(nn.Module):
    def __init__(self, backbone_model: nn.Module, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.backbone_model = backbone_model
        self.fc = nn.Linear(backbone_model.num_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, ppi):
        out = self.backbone_model.forward_features(ppi=ppi, flowstats=None)
        out = self.fc(out)
        out = self.bn(out)
        embeddings = F.normalize(out)
        return embeddings
