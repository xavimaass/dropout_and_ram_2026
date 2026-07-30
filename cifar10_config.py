from src.utils import make_dataset_cifar10
from src.jax_resnet.model import tanh

d_in, d_out, seed = 3072, 10, 42
N = None
classes = None          # set to e.g. [0, 1] for a 2-class subset experiment
X_train, Y_train, X_test, Y_test = make_dataset_cifar10(N=N, seed=seed, classes=classes)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

tau = 0.4 # longgood  0.4 #unstablelong: 0.8 # all: 0.4
n_steps = 25 #longgood 4_000 #25 #notrainemb: 50 #longrun: 1_000 #trainemb: 50
lr_in, lr_out = 0.4/d_in, 0.1*d_out #longgood: 0.4/d_in, 0.1*d_out
#unstable_long: 8.0/d_in, 0.5*d_out #notrainemb: 0.0, 0.0 
#longrun: 0.8/d_in, 0.5*d_out # train_emb: 0.1/d_in, 0.1*d_out
q = 0.5
batch_size = 64
last_particle_single_source = True
eval_every = 1 #long good 20 #longrun1 5 #all: 1
ACTIVATION = tanh
num_repetitions = 15 #longgood: 10 longrun1 #15 #all: 5
LOOP_SEED = 48

BASE_SETTING_STR = (
    f'_cifar10_tau{tau}_q{q}_nsteps{n_steps}_din{d_in}_dout{d_out}'
    f'_seed{seed}_N{N}_numrepetitions{num_repetitions}_loopseed{LOOP_SEED}'
    f'_batchsize{batch_size}_lpss{last_particle_single_source}_evalevery{eval_every}'
)