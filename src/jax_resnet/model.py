from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import random

from .dropout_masks import (
    sample_dropout_mask,
)
from .activations import relu, tanh


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Array = jax.Array
PRNGKey = Array


# ---------------------------------------------------------------------------
# Params dataclass
# ---------------------------------------------------------------------------
class FiniteResNetParams(NamedTuple):
    """
    All learnable parameters.

    Shapes
    ------
    W_in   : (D, n_in)   – input embedding
    W_out  : (D, n_out)  – output un-embedding
    U      : (L, M, D)   – first-layer weights of each 2LP unit
    V      : (L, M, D)   – second-layer weights of each 2LP unit
    """
    W_in:  Array   # (D, n_in)
    W_out: Array   # (D, n_out)
    U:     Array   # (L, M, D)
    V:     Array   # (L, M, D)


def init_params(
    key: PRNGKey,
    n_in: int,
    n_out: int,
    D: int,
    L: int,
    M: int,
    std_in: float = 1.0,
    std_out: float = 1.0,
    std_uv: float = 1.0,
) -> FiniteResNetParams:
    """Initialise all parameters with independent Gaussian entries."""
    k1, k2, k3, k4 = random.split(key, 4)
    W_in  = random.normal(k1, (D, n_in))  * (std_in/ jnp.sqrt(n_in)) # assuming fixed variance 1
    W_out = random.normal(k2, (D, n_out)) * (std_out) # assuming fixed variance 1
    U     = random.normal(k3, (L, M, D))  * (std_uv  * jnp.sqrt(D))
    V     = random.normal(k4, (L, M, D))  * (std_uv  * jnp.sqrt(D))
    return FiniteResNetParams(W_in=W_in, W_out=W_out, U=U, V=V)


# ---------------------------------------------------------------------------
# Unit  φ(z, h)  and  f^h(z, h)
# ---------------------------------------------------------------------------
def phi_2lp(u: Array, v: Array, h: Array, activation: Callable = relu) -> Array:
    """
    2LP unit:  φ((u, v), h) = v ⊙ ρ(u · h)
    u, v, h : (D,)  →  output : (D,)
    """
    D = u.shape[0]
    return v * activation(jnp.dot(u, h)/D)   # scalar times vector


def f_h(u: Array, v: Array, h: Array, activation: Callable = tanh) -> Array:
    """
    f^h(z, h) = diag(φ(z, h)) @ 1  =  φ(z, h)
    (applying diag(φ)·1 is equivalent to returning φ as a vector)
    """
    return phi_2lp(u, v, h, activation)


# ---------------------------------------------------------------------------
# Standard finite ResNet (no dropout)
# ---------------------------------------------------------------------------
def forward(
    params: FiniteResNetParams,
    x: Array,                          # (n_in,)
    activation: Callable = tanh,
) -> tuple[Array, Array]:
    """
    Standard forward pass.

    Returns
    -------
    h : (D,)   – final hidden state h(L, x)
    y : (n_out,) or scalar – model output ŷ(x)
    """
    L, M, D = params.U.shape

    # ---- layer 0 ----
    h = params.W_in @ x                # (D,)

    # ---- layers 1..L ----
    def layer_step(h, layer_idx):
        # sum over M units at depth l
        def unit_contrib(m):
            return f_h(params.U[layer_idx, m], params.V[layer_idx, m], h, activation)

        # vmap over M units
        contribs = jax.vmap(unit_contrib)(jnp.arange(M))   # (M, D)
        h_new = h + jnp.sum(contribs, axis=0) / (L * M)
        return h_new, None

    h, _ = jax.lax.scan(layer_step, h, jnp.arange(L))

    # ---- output ----
    y = (params.W_out.T @ h) / D       # (n_out,)
    return h, y


def forward_track(
    params: FiniteResNetParams,
    x: Array,                          # (n_in,)
    activation: Callable = tanh,
) -> tuple[Array, Array]:
    """
    Standard forward pass that tracks all hidden states.

    Returns
    -------
    h_all : (L+1, D)  – hidden states [h(0, x), h(1, x), ..., h(L, x)]
    y : (n_out,) or scalar – model output ŷ(x)
    """
    L, M, D = params.U.shape

    # ---- layer 0 ----
    h0 = params.W_in @ x               # (D,)

    # ---- layers 1..L ----
    def layer_step(h, layer_idx):
        def unit_contrib(m):
            return f_h(params.U[layer_idx, m], params.V[layer_idx, m], h, activation)

        contribs = jax.vmap(unit_contrib)(jnp.arange(M))   # (M, D)
        h_new = h + jnp.sum(contribs, axis=0) / (L * M)
        return h_new, h_new

    hL, h_hist = jax.lax.scan(layer_step, h0, jnp.arange(L))  # (L, D)
    h_all = jnp.concatenate([h0[None, :], h_hist], axis=0)    # (L+1, D)

    # ---- output ----
    y = (params.W_out.T @ hL) / D
    return h_all, y


# ---------------------------------------------------------------------------
# Dropout forward pass
# ---------------------------------------------------------------------------
def forward_dropout(
    params: FiniteResNetParams,
    x: Array,                          # (n_in,)
    mask: dict[str, Array],            # from sample_dropout_mask
    activation: Callable = tanh,
) -> tuple[Array, Array]:
    """
    Dropout forward pass.

    Returns
    -------
    h : (D,)    – final hidden state h^η(L, x)
    y : (n_out,) or scalar
    """
    L, M, D = params.U.shape
    eta_lmd  = mask["eta_lmd"]   # (L, M, D)
    eta_in   = mask["eta_in"]    # (D,)
    eta_out  = mask["eta_out"]   # (D,)

    # ---- layer 0 ----
    h = (1.0 + eta_in) * (params.W_in @ x)   # (D,)

    # ---- layers 1..L ----
    def layer_step(h, layer_idx):
        def unit_contrib(m):
            fh = f_h(params.U[layer_idx, m], params.V[layer_idx, m], h, activation)
            return fh * (1.0 + eta_lmd[layer_idx, m])          # (D,)

        contribs = jax.vmap(unit_contrib)(jnp.arange(M))   # (M, D)
        h_new = h + jnp.sum(contribs, axis=0) / (L * M)
        return h_new, None

    h, _ = jax.lax.scan(layer_step, h, jnp.arange(L))

    # ---- output ----
    y = (params.W_out.T @ ((1.0 + eta_out) * h)) / D   # (n_out,)
    return h, y


def forward_dropout_track(
    params: FiniteResNetParams,
    x: Array,                          # (n_in,)
    mask: dict[str, Array],            # from sample_dropout_mask
    activation: Callable = tanh,
) -> tuple[Array, Array]:
    """
    Dropout forward pass that tracks all hidden states.

    Returns
    -------
    h_all : (L+1, D)  – hidden states [h^η(0, x), h^η(1, x), ..., h^η(L, x)]
    y : (n_out,) or scalar
    """
    L, M, D = params.U.shape
    eta_lmd  = mask["eta_lmd"]   # (L, M, D)
    eta_in   = mask["eta_in"]    # (D,)
    eta_out  = mask["eta_out"]   # (D,)

    # ---- layer 0 ----
    h0 = (1.0 + eta_in) * (params.W_in @ x)   # (D,)

    # ---- layers 1..L ----
    def layer_step(h, layer_idx):
        def unit_contrib(m):
            fh = f_h(params.U[layer_idx, m], params.V[layer_idx, m], h, activation)
            return fh * (1.0 + eta_lmd[layer_idx, m])          # (D,)

        contribs = jax.vmap(unit_contrib)(jnp.arange(M))   # (M, D)
        h_new = h + jnp.sum(contribs, axis=0) / (L * M)
        return h_new, h_new

    hL, h_hist = jax.lax.scan(layer_step, h0, jnp.arange(L))  # (L, D)
    h_all = jnp.concatenate([h0[None, :], h_hist], axis=0)    # (L+1, D)

    # ---- output ----
    y = (params.W_out.T @ ((1.0 + eta_out) * hL)) / D
    return h_all, y


# ---------------------------------------------------------------------------
# Batched versions  (vmap over the input axis)
# ---------------------------------------------------------------------------
def batched_forward(
    params: FiniteResNetParams,
    X: Array,                          # (B, n_in)
    activation: Callable = tanh,
) -> tuple[Array, Array]:
    """Standard forward pass over a batch of inputs."""
    h_batch, y_batch = jax.vmap(
        lambda x: forward(params, x, activation)
    )(X)
    return h_batch, y_batch

def batched_forward_track(
    params: FiniteResNetParams,
    X: Array,                          # (B, n_in)
    activation: Callable = tanh,
) -> tuple[Array, Array]:
    """Standard forward pass that tracks all hidden states, over a batch of inputs."""
    h_batch, y_batch = jax.vmap(
        lambda x: forward_track(params, x, activation)
    )(X)
    return h_batch, y_batch

def batched_forward_dropout(
    params: FiniteResNetParams,
    X: Array,                          # (B, n_in)
    mask: dict[str, Array],
    activation: Callable = tanh,
) -> tuple[Array, Array]:
    """Dropout forward pass over a batch of inputs (shared mask across batch)."""
    h_batch, y_batch = jax.vmap(
        lambda x: forward_dropout(params, x, mask, activation)
    )(X)
    return h_batch, y_batch

def batched_forward_dropout_track(
    params: FiniteResNetParams,
    X: Array,                          # (B, n_in)
    mask: dict[str, Array],
    activation: Callable = tanh,
) -> tuple[Array, Array]:
    """Dropout forward pass that tracks all hidden states, over a batch of inputs (shared mask across batch)."""
    h_batch, y_batch = jax.vmap(
        lambda x: forward_dropout_track(params, x, mask, activation)
    )(X)
    return h_batch, y_batch

# ---------------------------------------------------------------------------
# Convenience: sample mask + dropout forward in one call
# ---------------------------------------------------------------------------
def stochastic_forward(
    key: PRNGKey,
    params: FiniteResNetParams,
    x: Array,
    q_layers: Array,
    q_in: float = 1.0,
    q_out: float = 1.0,
    activation: Callable = relu,
    internal_dropout_variant: str = "elementwise",
) -> tuple[Array, Array]:
    L, M, D = params.U.shape
    mask = sample_dropout_mask(
        key,
        L,
        M,
        D,
        q_layers,
        q_in,
        q_out,
        internal_variant=internal_dropout_variant,
    )
    return forward_dropout(params, x, mask, activation)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    key = random.PRNGKey(0)
    n_in, n_out, D, L, M = 8, 4, 16, 3, 5

    params = init_params(key, n_in, n_out, D, L, M)
    x = random.normal(random.PRNGKey(1), (n_in,))
    X = random.normal(random.PRNGKey(2), (6, n_in))

    # --- standard ---
    h, y = forward(params, x)
    print(f"[Standard]   h.shape={h.shape}, y.shape={y.shape}")

    h_b, y_b = batched_forward(params, X)
    print(f"[Batched]    h_b.shape={h_b.shape}, y_b.shape={y_b.shape}")

    # --- dropout ---
    q_layers = jnp.array([0.8, 0.9, 0.7])   # one keep-rate per layer
    mask = sample_dropout_mask(key, L, M, D, q_layers, q_in=0.9, q_out=0.9)
    print(f"[Mask]       eta_lmd.shape={mask['eta_lmd'].shape}, "
          f"eta_in.shape={mask['eta_in'].shape}")

    h_d, y_d = forward_dropout(params, x, mask)
    print(f"[Dropout]    h_d.shape={h_d.shape}, y_d.shape={y_d.shape}")

    h_db, y_db = batched_forward_dropout(params, X, mask)
    print(f"[Batch+Drop] h_db.shape={h_db.shape}, y_db.shape={y_db.shape}")

    # --- jit + grad ---
    @jax.jit
    def loss(params, X, mask):
        _, y = batched_forward_dropout(params, X, mask)
        return jnp.mean(y ** 2)

    grads = jax.grad(loss)(params, X, mask)
    print(f"[Grad]       dW_in.shape={grads.W_in.shape}, norm={jnp.linalg.norm(grads.W_in):.4f}")

    print("\nAll checks passed ✓")