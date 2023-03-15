# segmenting_dead_sea_scroll_fragments_for_a_scientific_image_set
This repository contains the source code and dataset used in the paper "Segmenting Dead Sea Scroll Fragments for a Scientific Image Set". 
The dataset include high-resolution images of Dead Sea Scroll fragments, along with corresponding ground-truth segmentation masks for evaluation purposes. The code and dataset can be used to reproduce the results of the paper, and can also serve as a starting point for further research in the field.

# Trained model

The trained model can be downloaded from [here](https://www.dropbox.com/s/5on3gy2c86t8tv9/model_final.pth?dl=0).

# Installation Guide

## Prerequisites

- CUDA 11.5
- Python 3.7

## Steps to Install

1. Check your CUDA version by typing the following command on your terminal:
    ```
    nvcc --version
    ```
   If your CUDA version is different than 11.5, consider updating the version or manually modifying the commands below to reflect the differences.

2. Create a virtual environment using conda with Python 3.7:
    ```
    conda create --name torchenv python=3.7
    ```

3. Install PyTorch 1.12.1, torchvision 0.13.1, and torchaudio 0.12.1, with CUDA 11.3:
    ```
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
    ```

4. Install OpenCV:
    ```
    conda install -c conda-forge opencv
    ```

5. Install JupyterLab:
    ```
    conda install jupyterlab
    ```

6. Install IPython kernel:
    ```
    conda install -c anaconda ipykernel
    ```

7. Install detectron2 from source using python pip:
    ```
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ```

8. Install setuptools:
    ```
    conda install setuptools==58.0.4
    ```

Once you have completed these installation steps, you should be ready to start using our project.
