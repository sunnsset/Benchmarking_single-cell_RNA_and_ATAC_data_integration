## scDART -- Learning latent embedding of multi-modalsingle cell data and cross-modality relationshipsimultaneously

scDART v0.1.0

[Zhang's Lab](https://xiuweizhang.wordpress.com), Georgia Institute of Technology

Developed by Ziqi Zhang, Chengkai Yang

## Description

**scDART** (**s**ingle **c**ell **D**eep learning model for **A**TAC-Seq and **R**NA-Seq **T**rajectory integration) is a scalable deep learning framework that embed the two data modalities of single cells, scRNA-seq and scATAC-seq data, into a shared low-dimensional latent space while preserving cell trajectory structures. Furthermore, **scDART** learns a nonlinear function represented by a neural network encoding the cross-modality relationship simultaneously when learning the latent space representations of the integrated dataset. 

The preprint is available on Genome Biology: [https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02706-x](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02706-x)


## Dependencies

```
Pytorch >= 1.5.0

numpy >= 1.18.2

scipy >= 1.4.1

pandas >= 1.0.3

sklearn >= 0.22.1

seaborn >= 0.10.0
```

## Installation

Clone the repository with

```
git clone https://github.com/PeterZZQ/scDART.git
```

And run 

```
pip install .
```

Uninstall using

```
pip uninstall scdart
```

## Usage

See `Example/demo.ipynb`.

<!-- ## Benchmark

See [https://github.com/PeterZZQ/scDART_bmk](https://github.com/PeterZZQ/scDART_bmk) for the benchmark result.

 -->


## Contents

* `scDART/` contains the python code for the package
* `data/` contains the sample simulated dataset. 
* `Example/` contains the demo code of scDART.

## Results in the preprint
The benchmark code, data and results are available through: [https:github.com/PeterZZQ/scDART_test](https:github.com/PeterZZQ/scDART_test) 

The script for data simulation can be found through: [https://github.com/PeterZZQ/Symsim2](https://github.com/PeterZZQ/Symsim2)

## Cite
**Zhang, Ziqi, Chengkai Yang, and Xiuwei Zhang. "Learning latent embedding of multi-modal single cell data and cross-modality relationship simultaneously." *bioRxiv* (2021).**