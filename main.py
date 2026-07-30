import pickle
import jax
import jax.numpy as jnp
from jax import random

from src.jax_resnet.model import init_params
from src.jax_resnet.training import train_scan_ce_jit, train_dropout_scan_ce_jit, train_ram_scan_ce_jit
from src.utils import align_tracked_particle_across_layers

DATASET = "cifar10"  # or "cifar10"

if DATASET == "mnist":
    from mnist_config import (
        d_in, d_out, seed, N, X_train, Y_train, X_test, Y_test, 
        tau, n_steps, lr_in, lr_out, q, batch_size, 
        last_particle_single_source, eval_every, ACTIVATION, num_repetitions, LOOP_SEED,
        BASE_SETTING_STR
        )
elif DATASET == "cifar10":
    from cifar10_config import (
    d_in, d_out, seed, N, X_train, Y_train, X_test, Y_test, 
    tau, n_steps, lr_in, lr_out, q, batch_size, 
    last_particle_single_source, eval_every, ACTIVATION, num_repetitions, LOOP_SEED,
    BASE_SETTING_STR
    )
else:
    raise ValueError(f"Unknown dataset: {DATASET}")

def loop_experiment(num_repetitions, loop_seed, train_fn, params0, kwargs, variants, dropout = True):
    final_params = {}
    histories = {}
    for variant in variants:
        print("Running variant:", variant)
        final_params_repeats = []
        histories_repeats = []
        for rep in range(num_repetitions):
            print(f"Repetition {rep+1}/{num_repetitions}")
            if dropout:
                fp, his = train_fn(
                    params0, 
                    internal_dropout_variant=variant, 
                    key=random.PRNGKey(loop_seed+rep),
                    **kwargs)
            else:
                fp, his = train_fn(
                    params0, 
                    key=random.PRNGKey(loop_seed+rep),
                    **kwargs)
                
            _ = jax.block_until_ready(his)
            final_params_repeats.append(fp)
            histories_repeats.append(his)
        final_params[variant] = final_params_repeats
        histories[variant] = histories_repeats
    return final_params, histories

def save_results(final_params_gd, histories_gd, final_params_do, histories_do,
                 final_params_ram, histories_ram, setting_str, data_dir):
    #os.makedirs(data_dir, exist_ok=True)
    with open(f'data/{data_dir}/final_params_gd_{setting_str}.pkl', 'wb') as f:
        pickle.dump(final_params_gd, f)
    with open(f'data/{data_dir}/histories_gd_{setting_str}.pkl', 'wb') as f:
        pickle.dump(histories_gd, f)
    with open(f'data/{data_dir}/final_params_do_{setting_str}.pkl', 'wb') as f:
        pickle.dump(final_params_do, f)
    with open(f'data/{data_dir}/histories_do_{setting_str}.pkl', 'wb') as f:
        pickle.dump(histories_do, f)
    with open(f'data/{data_dir}/final_params_ram_{setting_str}.pkl', 'wb') as f:
        pickle.dump(final_params_ram, f)
    with open(f'data/{data_dir}/histories_ram_{setting_str}.pkl', 'wb') as f:
        pickle.dump(histories_ram, f)

def main():
    D = 10 if DATASET == "mnist" else 50
    SHAPES = [
        # # (D,4,4),
        # # (D,4,8),
        # # (D,4,16),
        # # (D,4,32),
        # # (D,4,64),
        # # (D,4,128),
        # # (D,4,256),
        # # (D,4,512),
        # # (D,4,1024),
        # # (D,8,4),
        # (D,8,8),
        # (D,8,16),
        # (D,8,32),
        # (D,8,64),
        # (D,8,128),
        # (D,8,256),
        # (D,8,512),
        # # (D,8,1024),
        # # (D,16,4),
        # (D,16,8),
        # (D,16,16),
        # (D,16,32),
        # (D,16,64),
        # (D,16,128),
        # (D,16,256),
        # (D,16,512),
        # # (D,16,1024),
        # # (D,32,4),
        # (D,32,8),
        # (D,32,16),
        # (D,32,32),
        # (D,32,64),
        # (D,32,128),
        # (D,32,256),
        # (D,32,512),
        # # (D,32,1024),
        # # (D,64,4),
        # (D,64,8),
        # (D,64,16),
        # (D,64,32),
        # (D,64,64),
        # (D,64,128),
        # (D,64,256),
        # (D,64,512),
        # # (D,64,1024),
        # # (D,128,4),
        # (D,128,8),
        # (D,128,16),
        # (D,128,32),
        # (D,128,64),
        # (D,128,128),
        # (D,128,256),
        # (D,128,512),
        # # (D,128,1024),
        # # (D,256,4),
        # (D,256,8),
        # (D,256,16),
        # (D,256,32),
        # (D,256,64),
        # (D,256,128),
        # (D,256,256),
        # (D,256,512),
        # # (D,256,1024),
        # # (D,512,4),
        # (D,512,8),
        # (D,512,16),
        # (D,512,32),
        # (D,512,64),
        # (D,512,128),
        (D,512,256),
        (D,512,512),
        # # (D,512,1024),
        # # (D,1024,4),
        # # (D,1024,8),
        # # (D,1024,16),
        # # (D,1024,32),
        # # (D,1024,64),
        # # (D,1024,128),
        # # (D,1024,256),
        # # (D,1024,512),
        # # (D,1024,1024),
        ]

    for D,L,M in SHAPES:
        print(f"Running experiment with D={D}, L={L}, M={M}")
        VARML = 9*(M.bit_length()-1-2) + (L.bit_length()-1-2) # unique L and M identifier
        init_seed = seed + 44 + VARML
        # init_seed = seed + 44
        params0 = init_params(random.PRNGKey(init_seed), d_in, d_out, D, L, M)
        params0 = align_tracked_particle_across_layers(params0, particle_idx=-1)
        general_kwargs = dict(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, 
                        lr=tau, lr_in=lr_in, lr_out=lr_out, n_steps=n_steps, eval_every=eval_every, print_every=eval_every, 
                        batch_size=batch_size, track_outputs=False, activation=ACTIVATION)
        kwargs_do = general_kwargs | dict(q_layers=(q * jnp.ones(L)), q_in=1.0, q_out=1.0, 
                        single_source_last_particle=last_particle_single_source)
        variants = [
            "full_unit_dropout",
            "stochastic_depth",
            "single_source_M",
        ]
        loop_seed = LOOP_SEED + VARML  # NOT coupled across M and L realizations (does not change much)
        # loop_seed = LOOP_SEED
        print("Running GD")
        final_params_gd, histories_gd = loop_experiment(num_repetitions, loop_seed, train_scan_ce_jit, params0, general_kwargs, ["gd"], dropout=False)
        print("Running Dropout")
        final_params_do, histories_do = loop_experiment(num_repetitions, loop_seed, train_dropout_scan_ce_jit, params0, kwargs_do, variants, dropout=True)
        print("Running RAM")
        final_params_ram, histories_ram = loop_experiment(num_repetitions, loop_seed, train_ram_scan_ce_jit, params0, kwargs_do, variants, dropout=True)

        ### Save results from this run
        setting_str = f'L{L}_M{M}_D{D}' + BASE_SETTING_STR
        save_results(
            final_params_gd, histories_gd, 
            final_params_do, histories_do, 
            final_params_ram, histories_ram, 
            setting_str, data_dir=DATASET
            )

if __name__ == "__main__":
    main()        

