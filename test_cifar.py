import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
import plotting
from backends import get_generator, get_discriminator_cifar
from data_preprocessing import get_test_data_cifar, get_train_data_cifar, extract_features
from settings import get_settings
from model_utils import load_model
from validation_utils import hamming_dist
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist

# settings
args = get_settings()


# fixed random seeds
rng_data = np.random.RandomState(args.seed_data)
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))


# load CIFAR-10
trainx, trainy, txs, tys = get_train_data_cifar(args.data_dir, args.seed_data)
nr_batches_train = int(trainx.shape[0]/args.batch_size)


testx, testy = get_test_data_cifar(args.data_dir, reduce_to=1000, seed_data=6)


# specify generative model
gen_layers = get_generator(args.batch_size, theano_rng)
gen_dat = ll.get_output(gen_layers[-1])


# specify discriminative model
disc_layers,  f_low_dim, _ = get_discriminator_cifar(args.num_features)


load_model(gen_layers,args.generator_out)
load_model(disc_layers,args.discriminator_out)

x_temp = T.tensor4()

# Test generator in sampling procedure
samplefun = th.function(inputs=[],outputs=gen_dat)
sample_x = samplefun()
img_bhwc = np.transpose(sample_x[:100,], (0, 2, 3, 1))
img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
img = plotting.plot_img(img_tile, title='CIFAR10 samples')
plotting.plt.savefig("cifar_tgan_sample.png")

features = ll.get_output(disc_layers[-1], x_temp , deterministic=True)
generateTestF = th.function(inputs=[x_temp ], outputs=features)

samplefun = th.function(inputs=[],outputs=gen_dat)
sample_x = []
for k in range(5000):
    sample_x.append(samplefun())
sample_x = np.concatenate(sample_x, axis=0)

batch_size_test = 100
num_data_train = 10000
del disc_layers[-1]
print('Extracting features from test data')
test_features = extract_features(disc_layers, testx)
print('Extracting features from train data')
train_features = extract_features(disc_layers, trainx, args.batch_size)

sample_features = extract_features(disc_layers, sample_x, batch_size_test)

# calculating distances
test_features[test_features >=0.0] = 1
test_features[test_features < 0.0] = 0


train_features[train_features >=0.0] = 1
train_features[train_features < 0.0] = 0

sample_features[sample_features >= 0.0] = 1
sample_features[sample_features < 0.0] = 0

Y = hamming_dist(test_features,train_features)

ind = np.argsort(Y,axis=1)
prec = 0.0
acc = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
for k in range(np.shape(test_features)[0]):
    #class_values = testy[ind[k,:]]
    class_values = trainy[ind[k, :]]
    y_true = (testy[k] == class_values)
    y_scores = Y[k,ind[k,:]]
    y_scores = y_scores[::-1]
    ap = average_precision_score(y_true[0:num_data_train], y_scores[0:num_data_train])
    if not np.isnan(ap):
        prec = prec + ap
    for n in range(len(acc)):
        a = class_values[0:(n+1)]
        counts = np.bincount(a)
        b = np.where(counts==np.max(counts))[0]
        if testy[k] in b:
            acc[n] = acc[n] + (1.0/float(len(b)))
prec = prec/float(np.shape(test_features)[0])
acc= [x / float(np.shape(test_features)[0]) for x in acc]
print("Final results Hamming distance: ")
print("mAP value: %.4f "% prec)
'''for k in range(len(acc)):
    print("Accuracy for %d - NN: %.2f %%" % (k+1,100*acc[k]) )'''
