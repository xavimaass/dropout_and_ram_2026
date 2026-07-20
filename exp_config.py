from src.utils import make_dataset_mnist
from src.jax_resnet.model import relu, tanh
# SETTING
d_in, d_out, seed = 784, 2, 42
#N=10_000 # subsample of MNIST to use in the experiment (this affects the training time since we track the loss on the full test set over training iterations).
N=10_000
X_train, Y_train, X_test, Y_test = make_dataset_mnist(N=N, seed=seed, digits=[4,7])
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# Model parameters
tau = 0.4
n_steps = 200
lr_in, lr_out = 0.0, 0.0
q = 0.5
batch_size = 64
last_particle_single_source = True #Is the mask also shared in depth for the last particle?
eval_every = 1
ACTIVATION = tanh #relu
num_repetitions = 5
LOOP_SEED = 48

BASE_SETTING_STR = f'_tau{tau}_q{q}_nsteps{n_steps}_din{d_in}_dout{d_out}_seed{seed}_N{N}_numrepetitions{num_repetitions}_loopseed{LOOP_SEED}_batchsize{batch_size}_lpss{last_particle_single_source}_evalevery{eval_every}'