<p align="center">
    <img src="https://raw.githubusercontent.com/CESNET/cesnet-models/main/docs/images/models.svg" width="450">
</p>

[![](https://img.shields.io/badge/license-BSD-blue.svg)](https://github.com/CESNET/cesnet-models/blob/main/LICENCE)
[![](https://img.shields.io/badge/docs-cesnet--models-blue.svg)](https://cesnet.github.io/cesnet-models/)
[![](https://img.shields.io/badge/python->=3.10-blue.svg)](https://pypi.org/project/cesnet-models/)
[![](https://img.shields.io/pypi/v/cesnet-models)](https://pypi.org/project/cesnet-models/)


The goal of this project is to provide neural network architectures for traffic classification and their pre-trained weights.

The package provides two network architectures, 30pktTCNET and Multi-modal CESNET v2, both visualized in the following pictures. See the [getting started](https://cesnet.github.io/cesnet-models/getting_started/) page and [models](https://cesnet.github.io/cesnet-models/reference_models/) reference for more information.

:frog: :frog: See a related project [CESNET DataZoo](https://github.com/CESNET/cesnet-datazoo) providing large TLS and QUIC traffic datasets. :frog: :frog:

:notebook: :notebook: Example Jupyter notebooks are included in a separate [CESNET Traffic Classification Examples](https://github.com/CESNET/cesnet-tcexamples) repo. :notebook: :notebook:

### 30pktTCNET
<p align="center">
    <img src="https://raw.githubusercontent.com/CESNET/cesnet-models/main/docs/images/30pktTCNET.png" width="800">
</p>

### Multi-modal CESNET v2
<p align="center">
    <img src="https://raw.githubusercontent.com/CESNET/cesnet-models/main/docs/images/model-mm-cesnet-v2.png" width="400">
</p>

## Installation

Install the package from pip with:

```bash
pip install cesnet-models
```

or for editable install with:

```bash
pip install -e git+https://github.com/CESNET/cesnet-models
```

## Papers

Models from the following papers are included:

* [Fine-grained TLS services classification with reject option](https://doi.org/10.1016/j.comnet.2022.109467) <br>
Jan Luxemburk and Tomáš Čejka <br>
Computer Networks, 2023

* [Encrypted traffic classification: the QUIC case](https://doi.org/10.23919/TMA58422.2023.10199052) <br>
Jan Luxemburk and Karel Hynek <br>
2023 7th Network Traffic Measurement and Analysis Conference (TMA)