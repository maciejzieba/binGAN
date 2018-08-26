import lasagne
import numpy as np


def load_model(model, path):
    with np.load(path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    lasagne.layers.set_all_param_values(model, param_values)


def get_sim(fTrain, ftemp):
    ftemp = np.tile(ftemp, (fTrain.shape[0],1))
    dist = np.sqrt(np.sum((fTrain - ftemp)*(fTrain - ftemp),axis=1))
    ind = np.argsort(dist)
    return ind


def modify_indexes(ind, ordering):
    if ordering == 'easy_first':
        ind_out = ind[::-1]
    elif ordering == 'random':
        rng = np.random.RandomState()
        ind_out = ind[rng.permutation(len(ind))]
    else:
        ind_out = ind
    return ind_out