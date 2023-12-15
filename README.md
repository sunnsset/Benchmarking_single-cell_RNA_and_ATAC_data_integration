# Benchmarking multi-omics integration algorithms across single-cell RNA and ATAC data

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
*1469* is corresponding to dataset-T mentioned in the paper, *P0* is corresponding dataset-P, *uterus* is corresponding to dataset-U.
