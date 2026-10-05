# dropout_and_ram_2026

This repository implements the code for reproducing the experimental results for the paper "Dropout and Random Gradient Masking Are Asymptotically Equivalent in Large ResNets".

The repository is set up using [`uv`](https://docs.astral.sh/uv/). To run this code, you can install all required dependencies by simply running
```bash
uv sync
```

The folder `src` contains the source code for the ResNet implementation. 
The code for reproducing the result is organized in the following jupyter notebooks:
- `mnist_config.py` and `cifar10_config` contain all the parameters to set up the runs of the different experiments (respectively on the MNIST and CIFAR10 datasets).
- `main.py` contains the code for running the training dynamics of SGD-vanilla, SGD-Dropout and SGD-RaM on the relevant setting (for different configurations of (D, M, L)); and to save the results of the run. At the top of the file: define the DATASET and the mask_distribution. In the main() function: define the shapes to use in the experiment.
- `plots_from_multi_runs.ipynb` contains the code for creating a dataframe of "results" from the saved checkpoints of a training run.
- `plots_from_multi_runs_load.ipynb` contains the code for loading the "results" dataframe and producing the "convergence rate" plots displayed in the article.
- `plot_params_single_run.ipynb` and `plot_test_loss_single_run.ipynb` contain the code for loading a training checkpoint directly and producing (respectively) the plots for: the parameters at the end of training or the test_loss throughout training for the different variants.