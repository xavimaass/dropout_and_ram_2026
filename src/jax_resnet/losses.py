import jax
import jax.numpy as jnp
import jax.nn as jnn

Array = jax.Array

# LOSS
def quadratic_mean_error(y_hat: Array, y: Array):
    return (1/2) * jnp.mean((y_hat - y) ** 2)

def quadratic_mean_error_derivative(y_hat: Array, y: Array) -> Array:
    """Derivative of quadratic_mean_error w.r.t. y_hat: (1/n_out) * (y_hat - y)."""
    d_out = y_hat.shape[0]
    return (1.0 / d_out) * (y_hat - y)


def cross_entropy_from_logits(logits: Array, targets: Array) -> Array:
    """Mean cross-entropy from logits.

    Args:
        logits: shape (B, C) or (C,).
        targets: integer labels shape (B,) / scalar, or one-hot shape matching logits.
    """
    log_probs = jnn.log_softmax(logits, axis=-1)

    # Integer labels path.
    if targets.ndim == logits.ndim - 1:
        gather_idx = jnp.expand_dims(targets.astype(jnp.int32), axis=-1)
        per_sample = -jnp.take_along_axis(log_probs, gather_idx, axis=-1).squeeze(axis=-1)
        return jnp.mean(per_sample)

    # One-hot labels path.
    per_sample = -jnp.sum(targets * log_probs, axis=-1)
    return jnp.mean(per_sample)
