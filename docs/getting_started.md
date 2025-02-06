# Getting started

## Jupyter notebooks
Example Jupyter notebooks are provided at [https://github.com/CESNET/cesnet-tcexamples](https://github.com/CESNET/cesnet-tcexamples). Start with:

* Training of a neural network from scratch with data transformations - [example_train_nn.ipynb](https://nbviewer.org/github/CESNET/cesnet-tcexamples/blob/main/notebooks/example_train_nn.ipynb)
* Evaluate a pre-trained neural network I (TLS) - [reproduce_tls.ipynb](https://nbviewer.org/github/CESNET/cesnet-tcexamples/blob/main/notebooks/reproduce_tls.ipynb)
* Evaluate a pre-trained neural network II (QUIC) - [reproduce_quic.ipynb](https://nbviewer.org/github/CESNET/cesnet-tcexamples/blob/main/notebooks/reproduce_quic.ipynb)
* Multi-dataset evaluation of the 30pktTCNET_256 model - [cross_dataset_embedding_function.ipynb](https://nbviewer.org/github/CESNET/cesnet-tcexamples/blob/main/notebooks/cross_dataset_embedding_function.ipynb)

## Code snippets

```python
from cesnet_models.models import MM_CESNET_V2_Weights, mm_cesnet_v2

pretrained_weights = MM_CESNET_V2_Weights.CESNET_QUIC22_Week44
model = mm_cesnet_v2(weights=pretrained_weights, model_dir="models/")
```
This code will download pre-trained weights into the specified folder and initialize the mm-CESNET-v2 model. Available pre-trained weights for this model are listed in the MM_CESNET_V2_Weights enum and are named based on the dataset and training time period. When weights are specified, the model is returned in the [evaluation mode](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval). To train the model, you should first set it back in the training mode with `model.train()`.