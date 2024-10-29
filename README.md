# SGE2Hist
This repository contains the code implementation for the paper titled "SGE2Hist: Generating Histology Images from Spatially Resolved Single-cell Gene Expression via Cross-modal Generative Network"
SGE2Hist is a model designed to generate histological images based on single-cell gene expression data. The model leverages variational autoencoder techniques for low-dimensional representation of cell states and employs a novel latent space decoupling approach to improve clustering accuracy and capture microenvironment details.


## Requirements

- Python 3.9+
- PyTorch 2.4.1+
- NumPy 2.0.2+


## Dataset

The dataset used in this project is the 10X GENOMICS Visium HD Spatial Gene Expression Library. you can get from “https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-mouse-brain-fresh-frozen”  “https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-embryo”.

And the paired spatial resolution single-cell RNA sequencing and H&E images we processed can be got from "https://pan.baidu.com/s/1gZsRJmA4ChITDF-1_bAAFQ?pwd=1oob" for Mouse Brain 
"https://pan.baidu.com/s/13ippRbnTgow9eVtZcTwZNg?pwd=r233" for Mouse Embyro.


## License

This project is licensed under the MIT License. See the LICENSE file for details.
