# dropout_and_ram_2026

This repository implements the code for reproducing the experimental results for the paper "Dropout and Random Gradient Masking Are Asymptotically Equivalent in Large ResNets".

The repository is set up using [`uv`](https://docs.astral.sh/uv/). To run this code, you can install all required dependencies by simply running
```bash
uv sync
```

The folder `src` contains the source code for the ResNet implementation. 
The code for reproducing the result is organized in the following jupyter notebooks:
- `paper_experiments_mnist.ipynb` contains the code for running the training dynamics of SGD-vanilla, SGD-Dropout and SGD-RaM on the MNIST dataset; and to save the results of the run.
- `plots_from_single_run.ipynb` and `plots_from_multi_run.ipynb` contain the code for loading the obtained results and generating the figures displayed in the article.