# Available models
All models have the following behavior. When the `weights` parameter is specified, pre-trained weights will be downloaded and cached in the `model_dir` folder. The returned model will be in the [evaluation mode](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval).

## 30pktTCNET_256

An example of how to feed data into this model is provided in a Jupyter notebook with multi-dataset evaluation - [cross_dataset_embedding_function.ipynb](https://nbviewer.org/github/CESNET/cesnet-tcexamples/blob/main/notebooks/cross_dataset_embedding_function.ipynb).

::: models.model_30pktTCNET_256
    options:
        heading_level: 3
        show_root_heading: true
        separate_signature: true

## Multi-modal models

When the `weights` parameter is not specified, the model will be initialized with random weights and the following arguments become required:

* `num_classes` - the number of classes, which defines the output size of the last linear layer.
* `flowstats_input_size` - the number of flow statistics features and, therefore, the input size of the first linear layer processing them.
* `ppi_input_channels` - the number of channels in PPI sequences. The standard value is three for packet sizes, directions, and inter-arrival times.

### Input
Multi-modal models expect input in the format of `tuple(batch_ppi, batch_flowstats)`. The shapes are:

* batch_ppi `torch.tensor (B, ppi_input_channels, 30)` - batch size of `B` and  the length of PPI sequences is required to be 30.
* batch_flowstats `torch.tensor (B, flowstats_input_size)`

Jupyter notebooks listed on the [getting started][getting-started] page show how to feed data into multi-modal models.

::: models.mm_cesnet_v2
    options:
        heading_level: 3
        show_root_heading: true
        separate_signature: true

::: models.mm_cesnet_v1
    options:
        heading_level: 3
        show_root_heading: true
        separate_signature: true