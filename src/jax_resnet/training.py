import jax
import jax.numpy as jnp
from .dropout_masks import sample_dropout_mask, sample_zero_mask
from .model import batched_forward, batched_forward_dropout, batched_forward_track, batched_forward_dropout_track
from .losses import quadratic_mean_error, cross_entropy_from_logits
from .activations import tanh

def _maybe_debug_print(step_num, train_loss, noiseless_train_loss, test_loss, eval_every):
    def _print_fn(_):
        jax.debug.print(
            "step {step} | train_loss={train_loss:.6f}, (next) noiseless_train_loss={noiseless_train_loss:.6f}, (noiseless) test_loss={test_loss:.6f}",
            step=step_num,
            train_loss=train_loss,
            noiseless_train_loss=noiseless_train_loss,
            test_loss=test_loss,
        )
        return ()

    def _noop_fn(_):
        return ()

    jax.lax.cond(step_num % eval_every == 0, _print_fn, _noop_fn, operand=())


def _build_metrics(next_params, X_train, Y_train, X_test, Y_test, train_loss, track_forward_fn, track_outputs=True):
    return _build_metrics_with_loss(
        next_params,
        X_train,
        Y_train,
        X_test,
        Y_test,
        train_loss,
        track_forward_fn,
        quadratic_mean_error,
        track_outputs=track_outputs,
    )


def _build_metrics_with_loss(next_params, X_train, Y_train, X_test, Y_test, train_loss, track_forward_fn, loss_fn, track_outputs=True):
    h_train, out_train = track_forward_fn(next_params, X_train)
    h_test, out_test = track_forward_fn(next_params, X_test)
    noiseless_train_loss = loss_fn(out_train, Y_train)
    test_loss = loss_fn(out_test, Y_test)
    metrics = {
        "train_loss": train_loss,
        "noiseless_train_loss": noiseless_train_loss,
        "test_loss": test_loss,
    }
    if track_outputs:
        metrics.update({
            "train_output": out_train,
            "train_h": h_train,
            "test_output": out_test,
            "test_h": h_test,
        })
    return metrics


def make_metrics_fn(loss_fn, extra_metrics_fns=(), track_outputs=True):
    """Return a metrics function that matches the existing internal signature.

    `loss_fn` should take (model_output, targets) and return a scalar loss.
    `extra_metrics_fns` is an iterable of functions that accept
      (metrics, train_output, Y_train, test_output, Y_test) and return an
      updated metrics dict (used for adding accuracy for CE).
    `track_outputs` controls whether to include outputs and hidden activations.
    """
    def metrics_fn(params, X_train, Y_train, X_test, Y_test, train_loss, track_forward_fn):
        h_train, out_train = track_forward_fn(params, X_train)
        h_test, out_test = track_forward_fn(params, X_test)
        noiseless_train_loss = loss_fn(out_train, Y_train)
        test_loss = loss_fn(out_test, Y_test)
        metrics = {
            "train_loss": train_loss,
            "noiseless_train_loss": noiseless_train_loss,
            "test_loss": test_loss,
        }
        if track_outputs:
            metrics.update({
                "train_output": out_train,
                "train_h": h_train,
                "test_output": out_test,
                "test_h": h_test,
            })
        for fn in extra_metrics_fns:
            metrics = fn(metrics, out_train, Y_train, out_test, Y_test)
        return metrics

    return metrics_fn

def lr_scales_standard(L,M,D):
    return {
        'W_in': D,
        'W_out': D,
        'U': L * M * D,
        'V': L * M * D,
    }


def _resolve_lr(shared_lr, override_lr):
    return shared_lr if override_lr is None else override_lr


def _maybe_subsample(key, X, Y, batch_size):
    """Returns (X_batch, Y_batch). If batch_size is None, returns full dataset."""
    if batch_size is None:
        return X, Y
    n = X.shape[0]
    if batch_size >= n:
        return X, Y
    idx = jax.random.choice(key, n, shape=(batch_size,), replace=False)
    return X[idx], Y[idx]

def mse_loss(params, X, Y):
    """X: (B, n_in), Y: (B, n_out)"""
    _, Y_hat = batched_forward(params, X)
    return quadratic_mean_error(Y_hat, Y)

def mse_loss_dropout(params, X, Y, mask):
    _, Y_hat = batched_forward_dropout(params, X, mask)
    return quadratic_mean_error(Y_hat, Y)


def cross_entropy_loss(params, X, Y):
    """Cross entropy with model logits. X: (B, n_in), Y: (B,) or (B, n_classes)."""
    _, logits = batched_forward(params, X)
    return cross_entropy_from_logits(logits, Y)


def cross_entropy_loss_dropout(params, X, Y, mask):
    _, logits = batched_forward_dropout(params, X, mask)
    return cross_entropy_from_logits(logits, Y)


def classification_accuracy_from_logits(logits, targets):
    """Mean classification accuracy from logits and labels/one-hot targets."""
    pred = jnp.argmax(logits, axis=-1)
    if targets.ndim == logits.ndim:
        y_true = jnp.argmax(targets, axis=-1)
    else:
        y_true = targets.astype(jnp.int32)
    return jnp.mean((pred == y_true).astype(logits.dtype))


def _add_classification_metrics(metrics, train_logits, Y_train, test_logits, Y_test):
    return {
        **metrics,
        "train_accuracy": classification_accuracy_from_logits(train_logits, Y_train),
        "test_accuracy": classification_accuracy_from_logits(test_logits, Y_test),
    }

# GD STEP
def apply_custom_sgd(params, grads, lr, lr_in=None, lr_out=None, lr_U=None, lr_V=None):
    L, M, D = params.U.shape
    
    # Define scaling factors for each parameter
    lr_scales = lr_scales_standard(L, M, D)

    lr_in = _resolve_lr(lr, lr_in)
    lr_out = _resolve_lr(lr, lr_out)
    lr_U = _resolve_lr(lr, lr_U)
    lr_V = _resolve_lr(lr, lr_V)

    return params._replace(
        W_in=params.W_in - (lr_in * lr_scales["W_in"]) * grads.W_in,
        W_out=params.W_out - (lr_out * lr_scales["W_out"]) * grads.W_out,
        U=params.U - (lr_U * lr_scales["U"]) * grads.U,
        V=params.V - (lr_V * lr_scales["V"]) * grads.V,
    )


def gd_step(current_params, X_train, Y_train, lr, lr_in=None, lr_out=None, lr_U=None, lr_V=None):
    return gd_step_generic(current_params, X_train, Y_train, lr, mse_loss, mask=None, variant="standard", lr_in=lr_in, lr_out=lr_out, lr_U=lr_U, lr_V=lr_V)

def gd_step_dropout(current_params, X_train, Y_train, mask, lr, lr_in=None, lr_out=None, lr_U=None, lr_V=None):
    return gd_step_generic(current_params, X_train, Y_train, lr, mse_loss_dropout, mask=mask, variant="dropout", lr_in=lr_in, lr_out=lr_out, lr_U=lr_U, lr_V=lr_V)


def gd_step_ce(current_params, X_train, Y_train, lr, lr_in=None, lr_out=None, lr_U=None, lr_V=None):
    return gd_step_generic(current_params, X_train, Y_train, lr, cross_entropy_loss, mask=None, variant="standard", lr_in=lr_in, lr_out=lr_out, lr_U=lr_U, lr_V=lr_V)


def gd_step_dropout_ce(current_params, X_train, Y_train, mask, lr, lr_in=None, lr_out=None, lr_U=None, lr_V=None):
    return gd_step_generic(current_params, X_train, Y_train, lr, cross_entropy_loss_dropout, mask=mask, variant="dropout", lr_in=lr_in, lr_out=lr_out, lr_U=lr_U, lr_V=lr_V)


def _mask_ram_grads(grads, mask):
    eta_lmd = mask["eta_lmd"]
    eta_in = mask["eta_in"]
    eta_out = mask["eta_out"]

    return grads._replace(
        W_in=(1.0 + eta_in)[:, None] * grads.W_in,
        W_out=(1.0 + eta_out)[:, None] * grads.W_out,
        # RaM uses one mask value per (l, m); if eta varies across D, use the first coordinate.
        U=(1.0 + eta_lmd[..., 0])[..., None] * grads.U,
        V=(1.0 + eta_lmd[..., 0])[..., None] * grads.V,
    )


def gd_step_ram(current_params, X_train, Y_train, mask, lr, lr_in=None, lr_out=None, lr_U=None, lr_V=None):
    """Gradient step with RaM masking; mask must be a dict from sample_dropout_mask."""
    return gd_step_generic(current_params, X_train, Y_train, lr, mse_loss, mask=mask, variant="ram", lr_in=lr_in, lr_out=lr_out, lr_U=lr_U, lr_V=lr_V)


def gd_step_ram_ce(current_params, X_train, Y_train, mask, lr, lr_in=None, lr_out=None, lr_U=None, lr_V=None):
    """Cross-entropy gradient step with RaM masking."""
    return gd_step_generic(current_params, X_train, Y_train, lr, cross_entropy_loss, mask=mask, variant="ram", lr_in=lr_in, lr_out=lr_out, lr_U=lr_U, lr_V=lr_V)


# Generic GD STEP (unifies mse/ce, standard/dropout/ram)
def gd_step_generic(current_params, X, Y, lr, loss_fn, mask=None, variant="standard", lr_in=None, lr_out=None, lr_U=None, lr_V=None):
    """Generic gradient descent step.

    - `loss_fn` should be a callable with signature either
        (params, X, Y) -> loss  (for standard/ram)
      or
        (params, X, Y, mask) -> loss  (for dropout)
      Caller must pass the appropriate loss_fn for the variant.
    - `variant` in {"standard", "dropout", "ram"}.
    """
    if variant == "standard":
        train_loss, grads = jax.value_and_grad(loss_fn)(current_params, X, Y)
    elif variant == "dropout":
        if mask is None:
            raise ValueError("dropout variant requires a mask")
        # loss_fn is expected to accept mask as last arg
        wrapped = lambda p, Xb, Yb: loss_fn(p, Xb, Yb, mask)
        train_loss, grads = jax.value_and_grad(wrapped)(current_params, X, Y)
    elif variant == "ram":
        train_loss, grads = jax.value_and_grad(loss_fn)(current_params, X, Y)
        grads = _mask_ram_grads(grads, mask)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    next_params = apply_custom_sgd(current_params, grads, lr, lr_in, lr_out, lr_U, lr_V)
    return next_params, train_loss


gd_step_jit = jax.jit(gd_step)
gd_step_dropout_jit = jax.jit(gd_step_dropout)
gd_step_ram_jit = jax.jit(gd_step_ram)
gd_step_ce_jit = jax.jit(gd_step_ce)
gd_step_dropout_ce_jit = jax.jit(gd_step_dropout_ce)
gd_step_ram_ce_jit = jax.jit(gd_step_ram_ce)



# TRAIN LOOP
def train_scan(params, X_train, Y_train, X_test, Y_test, lr, n_steps, eval_every=10, lr_in=None, lr_out=None, lr_U=None, lr_V=None, activation=None):
    if activation is None:
        activation = tanh
    
    def track_forward(current_params, X):
        return batched_forward_track(current_params, X, activation)

    def step_fn(current_params, step_num):
        next_params, train_loss = gd_step(current_params, X_train, Y_train, lr, lr_in, lr_out, lr_U, lr_V)

        metrics = _build_metrics(next_params, X_train, Y_train, X_test, Y_test, train_loss, track_forward)
        _maybe_debug_print(step_num, train_loss, metrics["noiseless_train_loss"], metrics["test_loss"], eval_every)
        return next_params, metrics

    final_params, history = jax.lax.scan(
        step_fn,
        init=params,
        xs=jnp.arange(n_steps)
    )
    return final_params, history

train_scan_jit = jax.jit(train_scan, static_argnames=("n_steps",))


def train_scan_ce(params, X_train, Y_train, X_test, Y_test, lr, n_steps, eval_every=10, lr_in=None, lr_out=None, lr_U=None, lr_V=None, batch_size=None, key=None, metrics_on_batch=False, track_outputs=True, activation=None):
    """CE training with optional mini-batch SGD support."""
    if activation is None:
        activation = tanh
    
    metrics_fn = make_metrics_fn(cross_entropy_from_logits, extra_metrics_fns=(_add_classification_metrics,), track_outputs=track_outputs)
    
    def track_forward_with_activation(current_params, X):
        return batched_forward_track(current_params, X, activation)
    
    final_params, history = train_scan_generic(
        params,
        X_train,
        Y_train,
        X_test,
        Y_test,
        lr,
        n_steps,
        loss_fn_params=cross_entropy_loss,
        metrics_fn=metrics_fn,
        track_forward_fn=track_forward_with_activation,
        variant="standard",
        key=key,
        mask_sampler=None,
        batch_size=batch_size,
        eval_every=eval_every,
        metrics_on_batch=metrics_on_batch,
        lr_in=lr_in,
        lr_out=lr_out,
        lr_U=lr_U,
        lr_V=lr_V,
    )
    return final_params, history


train_scan_ce_jit = jax.jit(train_scan_ce, static_argnames=("n_steps", "batch_size", "metrics_on_batch", "track_outputs", "activation"))


def train_scan_generic(
    params,
    X_train,
    Y_train,
    X_test,
    Y_test,
    lr,
    n_steps,
    loss_fn_params,
    metrics_fn,
    track_forward_fn,
    variant="standard",
    key=None,
    mask_sampler=None,
    mask_args=(),
    mask_kwargs=None,
    batch_size=None,
    eval_every=10,
    metrics_on_batch=False,
    **lr_kwargs,
):
    """Generic training scan that handles standard / dropout / ram variants.

    - `loss_fn_params`: callable (params, X, Y) -> loss or (params, X, Y, mask) -> loss
    - `metrics_fn`: callable matching internal metrics signature
    - `track_forward_fn`: function used to compute noiseless outputs for metrics
    - `variant`: one of "standard", "dropout", "ram"
    - `mask_sampler`: if provided, called as mask_sampler(step_key, L, M, D, *mask_args, **mask_kwargs)
    - `key`: PRNGKey used to split per-step keys for mask sampling and/or mini-batch SGD
    - `batch_size`: if provided, subsample this many training points per step
    - `metrics_on_batch`: if True, compute metrics on the batch only; if False, compute on full dataset
    """
    L, M, D = params.U.shape

    needs_key = mask_sampler is not None or batch_size is not None
    if needs_key and key is None:
        raise ValueError("key required for mask_sampler or SGD batch_size")

    # Only split key if both mask_sampler and batch_size are used; otherwise use key directly
    if mask_sampler is not None and batch_size is not None:
        key_mask, key_batch = jax.random.split(key)
    elif mask_sampler is not None:
        key_mask = key
        key_batch = None
    elif batch_size is not None:
        key_mask = None
        key_batch = key
    else:
        key_mask, key_batch = None, None

    mask_keys = jax.random.split(key_mask, n_steps) if mask_sampler is not None else None
    batch_keys = jax.random.split(key_batch, n_steps) if batch_size is not None else None

    def step_fn(current_params, scan_inputs):
        mask_key, batch_key, step_num = scan_inputs

        X_batch, Y_batch = _maybe_subsample(batch_key, X_train, Y_train, batch_size)

        mask = (
            mask_sampler(mask_key, L, M, D, *mask_args, **(mask_kwargs or {}))
            if mask_sampler is not None
            else None
        )

        next_params, train_loss = gd_step_generic(
            current_params,
            X_batch,
            Y_batch,
            lr,
            loss_fn_params,
            mask=mask,
            variant=variant,
            **lr_kwargs,
        )

        # Compute metrics on batch or full dataset based on metrics_on_batch flag
        if metrics_on_batch:
            metrics = metrics_fn(next_params, X_batch, Y_batch, X_test, Y_test, train_loss, track_forward_fn)
        else:
            metrics = metrics_fn(next_params, X_train, Y_train, X_test, Y_test, train_loss, track_forward_fn)
        _maybe_debug_print(step_num, train_loss, metrics["noiseless_train_loss"], metrics["test_loss"], eval_every)
        return next_params, metrics

    dummy_keys = jnp.zeros((n_steps, 2), dtype=jnp.uint32)
    xs = (
        mask_keys if mask_keys is not None else dummy_keys,
        batch_keys if batch_keys is not None else dummy_keys,
        jnp.arange(n_steps),
    )

    final_params, history = jax.lax.scan(step_fn, init=params, xs=xs)
    return final_params, history


train_scan_generic_jit = jax.jit(
    train_scan_generic,
    static_argnames=("n_steps", "variant", "batch_size", "metrics_on_batch"),
)

def train_dropout_scan(
    params,
    X_train,
    Y_train,
    X_test,
    Y_test,
    lr,
    n_steps,
    q_layers,
    q_in,
    q_out,
    key,
    eval_every=10,
    lr_in=None,
    lr_out=None,
    lr_U=None,
    lr_V=None,
    internal_dropout_variant="elementwise",
    single_source_last_particle=False,
    batch_size=None,
    metrics_on_batch=False,
    track_outputs=True,
    activation=None,
):
    if activation is None:
        activation = tanh
    
    L, M, D = params.U.shape

    q_layers = jnp.array(q_layers)
    no_mask = sample_zero_mask(L, M, D)

    # track_forward for dropout should use a zero/noise mask
    def track_forward(current_params, X):
        return batched_forward_dropout_track(current_params, X, no_mask, activation)

    metrics_fn = make_metrics_fn(quadratic_mean_error, track_outputs=track_outputs)

    final_params, history = train_scan_generic(
        params,
        X_train,
        Y_train,
        X_test,
        Y_test,
        lr,
        n_steps,
        loss_fn_params=mse_loss_dropout,
        metrics_fn=metrics_fn,
        track_forward_fn=track_forward,
        variant="dropout",
        key=key,
        mask_sampler=sample_dropout_mask,
        mask_args=(q_layers, q_in, q_out),
        mask_kwargs={
            "internal_variant": internal_dropout_variant,
            "single_source_last_particle": single_source_last_particle,
        },
        batch_size=batch_size,
        eval_every=eval_every,
        metrics_on_batch=metrics_on_batch,
        lr_in=lr_in,
        lr_out=lr_out,
        lr_U=lr_U,
        lr_V=lr_V,
    )

    return final_params, history


train_dropout_scan_jit = jax.jit(
    train_dropout_scan,
    static_argnames=("n_steps", "internal_dropout_variant", "single_source_last_particle", "batch_size", "metrics_on_batch", "track_outputs", "activation"),
)


def train_dropout_scan_ce(
    params,
    X_train,
    Y_train,
    X_test,
    Y_test,
    lr,
    n_steps,
    q_layers,
    q_in,
    q_out,
    key,
    eval_every=10,
    lr_in=None,
    lr_out=None,
    lr_U=None,
    lr_V=None,
    internal_dropout_variant="elementwise",
    single_source_last_particle=False,
    batch_size=None,
    metrics_on_batch=False,
    track_outputs=True,
    activation=None,
):
    if activation is None:
        activation = tanh
    
    L, M, D = params.U.shape

    q_layers = jnp.array(q_layers)
    no_mask = sample_zero_mask(L, M, D)

    def track_forward(current_params, X):
        return batched_forward_dropout_track(current_params, X, no_mask, activation)

    metrics_fn = make_metrics_fn(cross_entropy_from_logits, extra_metrics_fns=(_add_classification_metrics,), track_outputs=track_outputs)

    final_params, history = train_scan_generic(
        params,
        X_train,
        Y_train,
        X_test,
        Y_test,
        lr,
        n_steps,
        loss_fn_params=cross_entropy_loss_dropout,
        metrics_fn=metrics_fn,
        track_forward_fn=track_forward,
        variant="dropout",
        key=key,
        mask_sampler=sample_dropout_mask,
        mask_args=(q_layers, q_in, q_out),
        mask_kwargs={
            "internal_variant": internal_dropout_variant,
            "single_source_last_particle": single_source_last_particle,
        },
        batch_size=batch_size,
        eval_every=eval_every,
        metrics_on_batch=metrics_on_batch,
        lr_in=lr_in,
        lr_out=lr_out,
        lr_U=lr_U,
        lr_V=lr_V,
    )

    return final_params, history


train_dropout_scan_ce_jit = jax.jit(
    train_dropout_scan_ce,
    static_argnames=("n_steps", "internal_dropout_variant", "single_source_last_particle", "batch_size", "metrics_on_batch", "track_outputs", "activation"),
)


def train_ram_scan(
    params,
    X_train,
    Y_train,
    X_test,
    Y_test,
    lr,
    n_steps,
    q_layers,
    key,
    q_in=1.0,
    q_out=1.0,
    eval_every=10,
    lr_in=None,
    lr_out=None,
    lr_U=None,
    lr_V=None,
    internal_dropout_variant="elementwise",
    single_source_last_particle=False,
    batch_size=None,
    metrics_on_batch=False,
    track_outputs=True,
    activation=None,
):
    if activation is None:
        activation = tanh
    
    L, M, D = params.U.shape

    q_layers = jnp.asarray(q_layers)

    metrics_fn = make_metrics_fn(quadratic_mean_error, track_outputs=track_outputs)
    
    def track_forward_with_activation(current_params, X):
        return batched_forward_track(current_params, X, activation)

    final_params, history = train_scan_generic(
        params,
        X_train,
        Y_train,
        X_test,
        Y_test,
        lr,
        n_steps,
        loss_fn_params=mse_loss,
        metrics_fn=metrics_fn,
        track_forward_fn=track_forward_with_activation,
        variant="ram",
        key=key,
        mask_sampler=sample_dropout_mask,
        mask_args=(q_layers, 1.0, 1.0),
        mask_kwargs={
            "internal_variant": internal_dropout_variant,
            "single_source_last_particle": single_source_last_particle,
        },
        batch_size=batch_size,
        eval_every=eval_every,
        metrics_on_batch=metrics_on_batch,
        lr_in=lr_in,
        lr_out=lr_out,
        lr_U=lr_U,
        lr_V=lr_V,
    )

    return final_params, history


train_ram_scan_jit = jax.jit(
    train_ram_scan,
    static_argnames=("n_steps", "internal_dropout_variant", "single_source_last_particle", "batch_size", "metrics_on_batch", "track_outputs", "activation"),
)


def train_ram_scan_ce(
    params,
    X_train,
    Y_train,
    X_test,
    Y_test,
    lr,
    n_steps,
    q_layers,
    key,
    q_in=1.0,
    q_out=1.0,
    eval_every=10,
    lr_in=None,
    lr_out=None,
    lr_U=None,
    lr_V=None,
    internal_dropout_variant="elementwise",
    single_source_last_particle=False,
    batch_size=None,
    metrics_on_batch=False,
    track_outputs=True,
    activation=None,
):
    if activation is None:
        activation = tanh
    
    L, M, D = params.U.shape

    q_layers = jnp.asarray(q_layers)

    metrics_fn = make_metrics_fn(cross_entropy_from_logits, extra_metrics_fns=(_add_classification_metrics,), track_outputs=track_outputs)
    
    def track_forward_with_activation(current_params, X):
        return batched_forward_track(current_params, X, activation)

    final_params, history = train_scan_generic(
        params,
        X_train,
        Y_train,
        X_test,
        Y_test,
        lr,
        n_steps,
        loss_fn_params=cross_entropy_loss,
        metrics_fn=metrics_fn,
        track_forward_fn=track_forward_with_activation,
        variant="ram",
        key=key,
        mask_sampler=sample_dropout_mask,
        mask_args=(q_layers, q_in, q_out),
        mask_kwargs={
            "internal_variant": internal_dropout_variant,
            "single_source_last_particle": single_source_last_particle,
        },
        batch_size=batch_size,
        eval_every=eval_every,
        metrics_on_batch=metrics_on_batch,
        lr_in=lr_in,
        lr_out=lr_out,
        lr_U=lr_U,
        lr_V=lr_V,
    )

    return final_params, history


train_ram_scan_ce_jit = jax.jit(
    train_ram_scan_ce,
    static_argnames=("n_steps", "internal_dropout_variant", "single_source_last_particle", "batch_size", "metrics_on_batch", "track_outputs", "activation"),
)
