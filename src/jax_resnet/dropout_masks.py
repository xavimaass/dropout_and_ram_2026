"""Dropout mask sampling utilities for the JAX ResNet implementation.

The internal LMD mask can be sampled in different structural variants while
sharing the same Bernoulli-rescaling convention:

- elementwise: independent mask for every (l, m, d)
- full_unit_dropout: shared across the D axis, so each unit is either kept or
  dropped as a whole
- stochastic_depth: shared across both M and D, so each residual branch is
  either kept or dropped as a whole
- residual_coordinate_dropout: shared across the M axis, so residual
    coordinates are dropped as a whole
- single_source_MD: one shared U(j, d) reused across all layers, creating
    depth-wise dependence
- single_source_D: one shared U(d) reused across all layers and M coordinates,
    creating depth-wise dependence and M-sharing
- single_source_M: one shared U(j) reused across all layers and D coordinates,
    creating depth-wise dependence and D-sharing
- single_source_full: one shared scalar U reused across all layers, M, and D,
    creating depth-wise dependence and full parameter-space sharing
"""
from typing import Literal

import jax
import jax.numpy as jnp
from jax import random

Array = jax.Array
PRNGKey = Array

InternalDropoutVariant = Literal[
    "elementwise",
    "full_unit_dropout",
    "stochastic_depth",
    "residual_coordinate_dropout",
    "single_source_MD",
    "single_source_D",
    "single_source_M",
    "single_source_full",
]


def _normalize_q_layers(q_layers: Array, L: int) -> Array:
    q_layers = jnp.asarray(q_layers)
    if q_layers.ndim == 0:
        q_layers = jnp.broadcast_to(q_layers, (L,))
    if q_layers.shape != (L,):
        raise ValueError(f"q_layers must have shape (L,) or be scalar; got {q_layers.shape}")
    return q_layers


def _sample_rescaled_mask(key: PRNGKey, q: Array | float, shape: tuple[int, ...]) -> Array:
    """Sample a Bernoulli-rescaled mask with keep-rate q. i.e.
        (1 - q) / q   with probability q
        -1            with probability 1 - q
    """
    q_arr = jnp.asarray(q, dtype=jnp.float32)
    b = random.bernoulli(key, p=q_arr, shape=shape).astype(jnp.float32)
    return jnp.where(b, (1.0 - q_arr) / q_arr, -1.0)


# Map variants to their base shape configuration (base_m, base_d)
# None means full dimension, 1 means shared across that dimension
_VARIANT_BASE_SHAPES = {
    "elementwise": (None, None),
    "full_unit_dropout": (None, 1),
    "stochastic_depth": (1, 1),
    "residual_coordinate_dropout": (1, None),
    "single_source_MD": (None, None),
    "single_source_D": (1, None),
    "single_source_M": (None, 1),
    "single_source_full": (1, 1),
}


def _single_source_tail_variant(variant: InternalDropoutVariant) -> InternalDropoutVariant:
    """Map a variant to the single-source behavior for a single M-particle tail.

    For the last particle, we preserve whether the variant is D-shared or
    D-independent, while enforcing depth-wise single-source coupling.
    """
    _, base_d = _VARIANT_BASE_SHAPES[variant]
    if base_d is None:
        # D-independent variants -> per-D shared uniform across layers.
        return "single_source_MD"
    # D-shared variants -> one shared scalar across D and layers.
    return "single_source_M"


def _sample_internal_dropout_mask_single_source_shared(
    key: PRNGKey,
    L: int,
    M: int,
    D: int,
    q_layers: Array,
    sample_shape: tuple[int, int],
) -> Array:
    """Sample single_source mask with shared uniform across layers.
    
    Samples once from a uniform matrix and reuses it across all layers,
    applying layer-specific thresholds via q_layers.
    """
    u = random.uniform(key, shape=sample_shape, dtype=jnp.float32)
    u_broadcast = jnp.broadcast_to(u, (M, D))
    q = q_layers.astype(jnp.float32)[:, None, None]
    return jnp.where(u_broadcast[None, :, :] <= q, (1.0 - q) / q, -1.0)


def sample_internal_dropout_mask(
    key: PRNGKey,
    L: int,
    M: int,
    D: int,
    q_layers: Array,
    variant: InternalDropoutVariant = "elementwise",
    single_source_last_particle: bool = False,
) -> Array:
    """Sample the internal LMD dropout mask with a chosen structural variant.

    Parameters
    ----------
    key:
        PRNG key.
    L, M, D:
        ResNet dimensions.
    q_layers:
        Keep rates for layers 1..L. Can be a scalar or an array of shape (L,).
    variant:
        Structural variant for the internal mask.
    single_source_last_particle:
        If True, sample the first ``M-1`` particles using ``variant`` and force
        the last particle to use the single-source equivalent implied by
        ``variant``. This avoids adding separate enum variants for the
        mixed-sampling regime.

    Returns
    -------
    eta_lmd : Array
        Mask of shape (L, M, D).
    """
    q_layers = _normalize_q_layers(q_layers, L)

    if single_source_last_particle:
        if M < 1:
            raise ValueError(f"M must be >= 1 when sampling dropout mask; got {M}")

        k_head, k_tail = random.split(key)

        if M == 1:
            eta_head = jnp.zeros((L, 0, D), dtype=jnp.float32)
        else:
            eta_head = sample_internal_dropout_mask(
                k_head,
                L,
                M - 1,
                D,
                q_layers,
                variant=variant,
                single_source_last_particle=False,
            )

        tail_variant = _single_source_tail_variant(variant)
        eta_tail = sample_internal_dropout_mask(
            k_tail,
            L,
            1,
            D,
            q_layers,
            variant=tail_variant,
            single_source_last_particle=False,
        )
        return jnp.concatenate([eta_head, eta_tail], axis=1)

    base_m, base_d = _VARIANT_BASE_SHAPES[variant]
    sample_shape = (
        M if base_m is None else base_m,
        D if base_d is None else base_d,
    )

    # For single_source variants, sample once and reuse across layers
    if variant.startswith("single_source"):
        return _sample_internal_dropout_mask_single_source_shared(key, L, M, D, q_layers, sample_shape)

    # For other variants, sample independently per layer
    def sample_layer(layer_idx: Array) -> Array:
        layer_key = random.fold_in(key, layer_idx)
        eta_l = _sample_rescaled_mask(layer_key, q_layers[layer_idx], sample_shape)
        return jnp.broadcast_to(eta_l, (M, D))

    return jax.vmap(sample_layer)(jnp.arange(L))


def sample_dropout_mask(
    key: PRNGKey,
    L: int,
    M: int,
    D: int,
    q_layers: Array,
    q_in: float,
    q_out: float,
    internal_variant: InternalDropoutVariant = "elementwise",
    single_source_last_particle: bool = False,
) -> dict[str, Array]:
    """Sample the full dropout mask dict used by the JAX ResNet model."""
    k1, k2, k3 = random.split(key, 3)
    eta_lmd = sample_internal_dropout_mask(
        k1,
        L,
        M,
        D,
        q_layers,
        internal_variant,
        single_source_last_particle=single_source_last_particle,
    )
    eta_in = _sample_rescaled_mask(k2, q_in, (D,))
    eta_out = _sample_rescaled_mask(k3, q_out, (D,))
    return {"eta_lmd": eta_lmd, "eta_in": eta_in, "eta_out": eta_out}


def sample_zero_mask(
    L: int,
    M: int,
    D: int,
    dtype: jnp.dtype = jnp.float32,
) -> dict[str, Array]:
    """Build a zero mask with the same structure as the sampled dropout mask."""
    return {
        "eta_lmd": jnp.zeros((L, M, D), dtype=dtype),
        "eta_in": jnp.zeros((D,), dtype=dtype),
        "eta_out": jnp.zeros((D,), dtype=dtype),
    }