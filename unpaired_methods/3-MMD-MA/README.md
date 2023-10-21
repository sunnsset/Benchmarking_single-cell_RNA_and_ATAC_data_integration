# README #

This is an implementation of MMD-MA in [Jointly embedding multiple single-cell omics measurements](https://www.biorxiv.org/content/10.1101/644310v1.full).

### Files ###
* manifoldAlignDistortionPen_mmd_multipleStarts.py: MMD-MA implementation
* X_linearkernel.tsv & Y_linearkernel.tsv: similarity matrices of Simulation1
* X2_linearkernel.tsv & Y2_linearkernel.tsv: similarity matrices of Simulation2
* X3_linearkernel.tsv & Y3_linearkernel.tsv: similarity matrices of Simulation3

### Usage ###
* `python manifoldAlignDistortionPen_mmd_multipleStarts.py [mat1] [mat2]`
* Hyperparameters can be set by optional parameters in command line
* For more detailed information, use `python manifoldAlignDistortionPen_mmd_multipleStarts.py -h`

### Examples ###
* Simulation1: `python manifoldAlignDistortionPen_mmd_multipleStarts.py X_linearkernel.tsv Y_linearkernel.tsv --l1 1e-6 --l2 1e-2 --p 5 --bandwidth 0.5 --seed 50`
* Simulation2: `python manifoldAlignDistortionPen_mmd_multipleStarts.py X2_linearkernel.tsv Y2_linearkernel.tsv --l1 1e-9 --l2 1e-7 --p 5 --bandwidth 0.1 --seed 50`
* Simualtion3: `python manifoldAlignDistortionPen_mmd_multipleStarts.py X3_linearkernel.tsv Y3_linearkernel.tsv --l1 1e-5 --l2 1e-6 --p 5 --bandwidth 1.2 --seed 50`