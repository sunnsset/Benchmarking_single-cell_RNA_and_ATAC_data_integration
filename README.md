# Benchmarking multi-omics integration algorithms across single-cell RNA and ATAC data

This is the code for paper: *Benchmarking multi-omics integration algorithms across single-cell RNA and ATAC data*.

doi: [https://doi.org/10.1101/2023.11.15.564963](https://doi.org/10.1093/bib/bbae095).

## Methods:
Each method is classified according to their category mentioned in the paper:

__paired methods__:

scMVP, MOFA+

__paired-guided methods__:

MultiVI, Cobolt

__unpaired methods__:

scDART, UnionCom, MMD-MA, scJoint, Harmony, Seurat v3, LIGER, GLUE

## Usage:
Within each method's folder, run the file *test_methodName-datasetName.ipynb* to do experiments.

## Metrics caculation:
Within the *test_accuracy* folder, run the file *test_accuracy.ipynb* to calculate detailed metrics and visualize the result.

## Dataset names:
*1469* is corresponding to dataset-T mentioned in the paper, *P0* is corresponding to dataset-P, *uterus* is corresponding to dataset-U.

The related data could be downloaded from https://www.alipan.com/s/AYbCNy1WPgj.
