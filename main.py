import pickle
import jax
import jax.numpy as jnp
from jax import random

from src.jax_resnet.model import init_params
from src.jax_resnet.training import train_scan_ce_jit, train_dropout_scan_ce_jit, train_ram_scan_ce_jit
from src.utils import align_tracked_particle_across_layers

from exp_config import (
    d_in, d_out, seed, N, X_train, Y_train, X_test, Y_test, 
    tau, n_steps, lr_in, lr_out, q, batch_size, 
    last_particle_single_source, eval_every, ACTIVATION, num_repetitions, LOOP_SEED,
    BASE_SETTING_STR
    )

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

def save_results(final_params_gd, histories_gd, final_params_do, histories_do, final_params_ram, histories_ram, setting_str):
    # Export (serialize) to pickle
    with open(f'data/mnist/final_params_gd_{setting_str}.pkl', 'wb') as f:   # 'wb' = write binary
        pickle.dump(final_params_gd, f)

    with open(f'data/mnist/histories_gd_{setting_str}.pkl', 'wb') as f:   # 'wb' = write binary
        pickle.dump(histories_gd, f)

    with open(f'data/mnist/final_params_do_{setting_str}.pkl', 'wb') as f:   # 'wb' = write binary
        pickle.dump(final_params_do, f)

    with open(f'data/mnist/histories_do_{setting_str}.pkl', 'wb') as f:   # 'wb' = write binary
        pickle.dump(histories_do, f)

    with open(f'data/mnist/final_params_ram_{setting_str}.pkl', 'wb') as f:   # 'wb' = write binary
        pickle.dump(final_params_ram, f)

    with open(f'data/mnist/histories_ram_{setting_str}.pkl', 'wb') as f:   # 'wb' = write binary
        pickle.dump(histories_ram, f)

def main():

    SHAPES = [
        (10,4,4),
        (10,4,8),
        (10,4,16),
        (10,4,32),
        (10,4,64),
        (10,4,128),
        (10,4,256),
        (10,4,512),
        (10,4,1024),
        (10,8,4),
        (10,8,8),
        (10,8,16),
        (10,8,32),
        (10,8,64),
        (10,8,128),
        (10,8,256),
        (10,8,512),
        (10,8,1024),
        (10,16,4),
        (10,16,8),
        (10,16,16),
        (10,16,32),
        (10,16,64),
        (10,16,128),
        (10,16,256),
        (10,16,512),
        (10,16,1024),
        (10,32,4),
        (10,32,8),
        (10,32,16),
        (10,32,32),
        (10,32,64),
        (10,32,128),
        (10,32,256),
        (10,32,512),
        (10,32,1024),
        (10,64,4),
        (10,64,8),
        (10,64,16),
        (10,64,32),
        (10,64,64),
        (10,64,128),
        (10,64,256),
        (10,64,512),
        (10,64,1024),
        (10,128,4),
        (10,128,8),
        (10,128,16),
        (10,128,32),
        (10,128,64),
        (10,128,128),
        (10,128,256),
        (10,128,512),
        (10,128,1024),
        (10,256,4),
        (10,256,8),
        (10,256,16),
        (10,256,32),
        (10,256,64),
        (10,256,128),
        (10,256,256),
        (10,256,512),
        (10,256,1024),
        (10,512,4),
        (10,512,8),
        (10,512,16),
        (10,512,32),
        (10,512,64),
        (10,512,128),
        (10,512,256),
        (10,512,512),
        (10,512,1024),
        (10,1024,4),
        (10,1024,8),
        (10,1024,16),
        (10,1024,32),
        (10,1024,64),
        (10,1024,128),
        (10,1024,256),
        (10,1024,512),
        (10,1024,1024),
        ]

    for D,L,M in SHAPES:
        print(f"Running experiment with D={D}, L={L}, M={M}")

        params0 = init_params(random.PRNGKey(seed + 44), d_in, d_out, D, L, M)
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
        #loop_seed = LOOP_SEED + 9*(M.bit_length()-1-2) + (L.bit_length()-1-2) # NOT coupled across M and L realizations (does not change much)
        loop_seed = LOOP_SEED
        print("Running GD")
        final_params_gd, histories_gd = loop_experiment(num_repetitions, loop_seed, train_scan_ce_jit, params0, general_kwargs, ["gd"], dropout=False)
        print("Running Dropout")
        final_params_do, histories_do = loop_experiment(num_repetitions, loop_seed, train_dropout_scan_ce_jit, params0, kwargs_do, variants, dropout=True)
        print("Running RAM")
        final_params_ram, histories_ram = loop_experiment(num_repetitions, loop_seed, train_ram_scan_ce_jit, params0, kwargs_do, variants, dropout=True)

        ### Save results from this run
        setting_str = f'L{L}_M{M}_D{D}' + BASE_SETTING_STR
        save_results(final_params_gd, histories_gd, final_params_do, histories_do, final_params_ram, histories_ram, setting_str)

if __name__ == "__main__":
    main()        

