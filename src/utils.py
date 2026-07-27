import jax
import jax.numpy as jnp
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torchvision import datasets as tv_datasets

from src.jax_resnet.model import FiniteResNetParams, batched_forward_track

def make_dataset_mnist(N=None, seed=42, digits=None):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, Y = mnist.data.astype(np.float32) / 255.0, mnist.target.astype(np.int32)

    # Filter to chosen digits
    if digits is not None:
        mask = np.isin(Y, digits)
        X, Y = X[mask], Y[mask]
        # Remap labels to 0, 1, 2, ... based on position in digits list
        label_map = {d: i for i, d in enumerate(digits)}
        Y = np.array([label_map[y] for y in Y], dtype=np.int32)

    if N is not None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))[:N]
        X, Y = X[idx], Y[idx]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=seed
    )

    n_classes = len(digits) if digits is not None else 10
    Y_train_oh = jax.nn.one_hot(Y_train, n_classes)
    Y_test_oh  = jax.nn.one_hot(Y_test,  n_classes)

    return jnp.array(X_train), jnp.array(Y_train_oh), jnp.array(X_test), jnp.array(Y_test_oh)


def align_tracked_particle_across_layers(params, particle_idx=-1) -> FiniteResNetParams:
        """Make U[:, particle_idx, :] and V[:, particle_idx, :] identical across layers."""
        u_ref = params.U[0, particle_idx, :]  # (D,)
        v_ref = params.V[0, particle_idx, :]  # (D,)

        U_new = params.U.at[:, particle_idx, :].set(jnp.broadcast_to(u_ref, params.U[:, particle_idx, :].shape))
        V_new = params.V.at[:, particle_idx, :].set(jnp.broadcast_to(v_ref, params.V[:, particle_idx, :].shape))
        return FiniteResNetParams(W_in=params.W_in, W_out=params.W_out, U=U_new, V=V_new)


def compute_train_h_out(params, X_test, activation):
    th, ty = batched_forward_track(params, X_test, activation=activation)
    return th, ty


def add_train_h_out(params_dict, history_dict, X_test, activation):
    new_dict = history_dict.copy()
    for variant in params_dict.keys():
        for j in range(len(params_dict[variant])):
            th, ty = compute_train_h_out(params_dict[variant][j], X_test, activation)
            new_dict[variant][j]["test_h"] = th
            new_dict[variant][j]["test_output"] = ty
    return new_dict


def make_dataset_cifar10(N=None, seed=42, classes=None, root='./datasets/cifar10'):
    """Load CIFAR-10, cache under `root`, and return JAX arrays."""
    train = tv_datasets.CIFAR10(root=root, train=True, download=True)
    test = tv_datasets.CIFAR10(root=root, train=False, download=True)

    # HWC uint8 -> float in [0,1]
    X_train = np.array(train.data, dtype=np.float32) / 255.0
    Y_train = np.array(train.targets, dtype=np.int32)
    X_test = np.array(test.data, dtype=np.float32) / 255.0
    Y_test = np.array(test.targets, dtype=np.int32)

    # Optional per-channel standardization using training statistics
    mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Flatten to vectors for the FC model
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Filter to a subset of classes and remap labels
    if classes is not None:
        mask_tr = np.isin(Y_train, classes)
        X_train, Y_train = X_train[mask_tr], Y_train[mask_tr]
        mask_te = np.isin(Y_test, classes)
        X_test, Y_test = X_test[mask_te], Y_test[mask_te]
        label_map = {c: i for i, c in enumerate(classes)}
        Y_train = np.array([label_map[y] for y in Y_train], dtype=np.int32)
        Y_test = np.array([label_map[y] for y in Y_test], dtype=np.int32)

    if N is not None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_train))[:N]
        X_train, Y_train = X_train[idx], Y_train[idx]

    n_classes = len(classes) if classes is not None else 10
    Y_train_oh = jax.nn.one_hot(Y_train, n_classes)
    Y_test_oh = jax.nn.one_hot(Y_test, n_classes)

    return jnp.array(X_train), jnp.array(Y_train_oh), jnp.array(X_test), jnp.array(Y_test_oh)