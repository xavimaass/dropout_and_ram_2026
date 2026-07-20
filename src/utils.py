import jax
import jax.numpy as jnp
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

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