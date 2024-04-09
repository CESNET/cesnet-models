# Available models
Functions for creating the models ([`mm_cesnet_v1`][models.mm_cesnet_v1] and [`mm_cesnet_v2`][models.mm_cesnet_v2]) share the following behavior. When the `weights` parameter is specified, the pre-trained weights will be downloaded and cached in the `model_dir` folder. The returned model will be in the [evaluation mode](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval).

On the other hand, when the `weights` parameter is not specified, the model will be initialized with random weights (default PyTorch behavior), and the following arguments become mandatory:

* `num_classes` - the number of classes, which defines the output size of the last linear layer.
* `flowstats_input_size` - the number of flow statistics features and, therefore, the input size of the first linear layer processing them.
* `ppi_input_channels` - the number of channels in PPI sequences. The standard value is three for packet sizes, directions, and inter-arrival times.

### Input
The models expect input in the format of `tuple(batch_ppi, batch_flowstats)`. The shapes are:

* batch_ppi `torch.tensor (B, ppi_input_channels, 30)` - batch size of `B` and  the length of PPI sequences is required to be 30.
* batch_flowstats `torch.tensor (B, flowstats_input_size)`

All Jupyter notebooks listed on the [getting started][getting-started] page show how to feed data into the models.

::: models.mm_cesnet_v2
    options:
        heading_level: 2
        show_root_heading: true
        separate_signature: true

::: models.mm_cesnet_v1
    options:
        heading_level: 2
        show_root_heading: true
        separate_signature: true