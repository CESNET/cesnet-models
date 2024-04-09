<p align="center">
    <img src="https://raw.githubusercontent.com/CESNET/cesnet-models/main/docs/images/models.svg" width="450">
</p>

[![](https://img.shields.io/badge/license-BSD-blue.svg)](https://github.com/CESNET/cesnet-models/blob/main/LICENCE)
[![](https://img.shields.io/badge/docs-cesnet--models-blue.svg)](https://cesnet.github.io/cesnet-models/)
[![](https://img.shields.io/badge/python->=3.10-blue.svg)](https://pypi.org/project/cesnet-models/)
[![](https://img.shields.io/pypi/v/cesnet-models)](https://pypi.org/project/cesnet-models/)


The goal of this project is to provide neural network architectures for traffic classification and their pre-trained weights. The weights were trained using public datasets available in the [CESNET DataZoo](https://github.com/CESNET/cesnet-datazoo) package.

The newest network architecture is named Multi-modal CESNET v2 (mm-CESNET-v2) and is visualized in the following picture. See the [getting started](https://cesnet.github.io/cesnet-models/getting_started/) page and [models](https://cesnet.github.io/cesnet-models/reference_models/) reference for more information.

:frog: :frog: See a related project [CESNET DataZoo](https://github.com/CESNET/cesnet-datazoo) providing three large network traffic datasets. :frog: :frog:

:notebook: :notebook: Example Jupyter notebooks are included in a separate [CESNET Traffic Classification Examples](https://github.com/CESNET/cesnet-tcexamples) repo. :notebook: :notebook:

### Multi-modal CESNET v2
<p align="center">
    <img src="https://raw.githubusercontent.com/CESNET/cesnet-models/main/docs/images/model-mm-cesnet-v2.png" width="450">
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