import jax
import jax.numpy as jnp

Array = jax.Array

# ---------------------------------------------------------------------------
# Activation function
# ---------------------------------------------------------------------------
def relu(x: Array) -> Array:
    return jnp.maximum(0.0, x)

def relu_derivative(x: Array) -> Array:
    return (x > 0.0).astype(x.dtype)

def tanh(x: Array) -> Array:
    return jnp.tanh(x)

def tanh_derivative(x: Array) -> Array:
    """Derivative of tanh: 1 / cosh^2(x) = 1 - tanh^2(x)."""
    return 1.0 - jnp.tanh(x) ** 2