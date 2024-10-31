# SGE2Hist
This repository contains the code implementation for the paper titled "SGE2Hist: Generating Histology Images from Spatially Resolved Single-cell Gene Expression via Cross-modal Generative Network"
SGE2Hist is a model designed to generate histological images based on single-cell gene expression data. 

## Requirements

- Python 3.9+
- PyTorch 2.4.1+
- NumPy 2.0.2+


## Dataset

The dataset used in this project is the 10X GENOMICS Visium HD Spatial Gene Expression Library. you can get from “https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-mouse-brain-fresh-frozen”  “https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-embryo”.

And the paired spatial resolution single-cell RNA sequencing and H&E images we processed can be got from "https://pan.baidu.com/s/1gZsRJmA4ChITDF-1_bAAFQ?pwd=1oob" for Mouse Brain 
"https://pan.baidu.com/s/13ippRbnTgow9eVtZcTwZNg?pwd=r233" for Mouse Embyro.


## Usage



To use the SGE2Hist model, follow these steps:


1. **Pre-train the VAE**: First, you need to pre-train the VAE architecture to determine the initial Gaussian mixture distribution for clustering cell types. Use the following command:

    ```bash
    python pretrain.py 
    ```
2. **Load the Pre-trained VAE**: After pre-training, load the pre-trained VAE model for further training:"https://pan.baidu.com/s/1YxcC95lApwpzi27LoUBldw?pwd=wodz"

    ```python
    from SGE2Hist import SGE2Hist
    from Dataset import DataSet

    # Initialize the SGE2Hist model with the pre-trained VAE
    model = SGE2Hist()
    model.load_state_dict(torch.load('Pretrained weight/pre_embyro.pt'))

    # Load the dataset
    dataset = DataSet(datadir='', transform=transform)

    # Train the complete model
    model.train(dataset)
    ```

**Pre-trained Model Weights**：You can download the pre-trained model weights from: 
## License

This project is licensed under the MIT License. See the LICENSE file for details.
