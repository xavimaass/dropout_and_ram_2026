from src.utils import make_dataset_cifar10
from src.jax_resnet.model import tanh

d_in, d_out, seed = 3072, 10, 42
N = None
classes = None          # set to e.g. [0, 1] for a 2-class subset experiment
X_train, Y_train, X_test, Y_test = make_dataset_cifar10(N=N, seed=seed, classes=classes)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

tau = 0.4
n_steps = 50
lr_in, lr_out = 0.1/d_in, 0.1*d_out # 0.1/d_in, 0.1*d_out
q = 0.5
batch_size = 64
last_particle_single_source = True
eval_every = 1
ACTIVATION = tanh
num_repetitions = 5
LOOP_SEED = 48

BASE_SETTING_STR = (
    f'_cifar10_tau{tau}_q{q}_nsteps{n_steps}_din{d_in}_dout{d_out}'
    f'_seed{seed}_N{N}_numrepetitions{num_repetitions}_loopseed{LOOP_SEED}'
    f'_batchsize{batch_size}_lpss{last_particle_single_source}_evalevery{eval_every}'
)