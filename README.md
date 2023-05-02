# LapMMIRF : Multi-Modal Image Registration Framework using Laplacian Pyramid
Multi-Modal Medical Image Registration Framework using Laplacian Pyramid for precise image alignment and analysis.

## Prerequisites
- `Python 3.9.16+`
- `Pytorch 1.7.0 - 2.0.0`
- `NumPy`
- `NiBabel`

This code has been tested on [Google Colaboraty](https://colab.research.google.com/) with `Pytorch 2.0.0+cu118` and NVIDIA A100-SXM4-40GB GPU.

## Instructions
All required imports and functions are contained in the `LapMMIRF.ipynb` file, running all cells would do the following. 
- `download necessary datasets from the IXI dataset`
- `install and import standard and third part libraries`
- `train the model using the specified hyperparameters`
- `save the trained model to the output folder as model.pth`
- `test the model with random images and saving the registered images`

## Dataset
All fixed and moving images in this directory is obtained from the [IXI Dataset](https://brain-development.org/ixi-dataset/). 
- Fixed Image: PD MRI Scans
- Moving Image: T2-Weighted MRI Scans


###### Keywords
Keywords: Deep Learning, Multi-Modal, Image Registration, Large Deformation, Laplacian Pyramid Networks, Convolutional Neural Networks
