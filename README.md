# BuDDI

This repository contains all the model code used in [**BuDDI: Bulk Deconvolution with Domain Invariance to predict cell-type-specific perturbations from bulk**](https://www.biorxiv.org/content/10.1101/2023.07.20.549951v1)

A tutorial, as well as all code needed to reproduce the figures from the manuscript, are provided in this companion repository: https://github.com/greenelab/buddi_analysis

### Overview
While single-cell experiments provide deep cellular resolution within a single sample, some are inherently more challenging than bulk experiments due to dissociation difficulties, cost, or limited tissue availability. 
This creates a situation where we have deep cellular profiles of one sample or condition, and bulk profiles across multiple samples and conditions. 

A schematic of what BuDDi's methodological goal is shown below:

<p align="center">
<img src="https://github.com/nrosed/buddi/blob/main/buddi_goal.png" width="50%" height="50%">
</p>
  
**To bridge this gap, we propose BuDDI (BUlk Deconvolution with Domain Invariance).**

BuDDI utilizes domain adaptation techniques to effectively integrate available corpora of case-control bulk and reference scRNA-seq observations to infer cell-type-specific perturbation effects. 
BuDDI achieves this by learning independent latent spaces within a single variational autoencoder (VAE) encompassing at least four sources of variability: 1) cell-type proportion, 2) perturbation effect, 3) structured experimental variability, and 4) remaining variability. 
Since each latent space is encouraged to be independent, we simulate perturbation responses by independently composing each latent space to simulate cell-type-specific perturbation responses.

Below is a schematic of the VAE structure of BuDDI:
![BuDDI schematic](https://www.biorxiv.org/content/biorxiv/early/2023/07/22/2023.07.20.549951/F1.large.jpg?width=800&height=600&carousel=1)

### Installation

First, you need to install dependencies using your package manager. Below is an example for OS X using brew.

```bash
brew install hdf5 c-blosc cython
export HDF5_DIR=/opt/homebrew/opt/hdf5 
export BLOSC_DIR=/opt/homebrew/opt/c-blosc
```

Below is an example for Ubuntu using apt-get.

```bash
apt-get install libhdf5-serial-dev
apt-get install libblosc-dev
apt-get install pip install typing-extensions --upgrade
```

Second, you will install BuDDI from github using pip. It is recommended that you install this into a virtual env that you can activate before running the provided analysis scripts or your own BuDDI experiments.

```bash
pip install git+https://github.com/greenelab/buddi.git
```

### Usage
See the [tutorial](https://github.com/greenelab/buddi_analysis) for detailed instructions on how to use BuDDI.

To import BuDDI and its helper methods in your notebook, simply add the following

```python
from buddi import buddi
from buddi.preprocessing import sc_preprocess
from buddi.plotting import validation_plotting as vp
```
