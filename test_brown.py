import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
import plotting
from backends import get_generator, get_discriminator_brown
from data_preprocessing import get_data_patches_test, extract_features, get_data_patches_training
from settings import get_settings
from model_utils import load_model
from validation_utils import hamming_dist
from eval import fp_at_95, roc



# settings
args = get_settings()
print("Testing .........")
print("Trained on: " +args.data_name)
print("Test on: " + args.test_data)
# fixed random seeds
rng_data = np.random.RandomState(args.seed_data)
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))


# load CIFAR-10
test_matched, test_unmatched = get_data_patches_test(args.test_data, args.data_dir)

trainx = get_data_patches_training(args.data_name, args.data_dir)
trainx = trainx[rng.permutation(trainx.shape[0])]

# specify generative model
gen_layers = get_generator(args.batch_size, theano_rng)
gen_dat = ll.get_output(gen_layers[-1])


# specify discriminative model
disc_layers, f_low_dim, _ = get_discriminator_brown(args.num_features)

load_model(gen_layers,args.generator_out)
load_model(disc_layers,args.discriminator_out)

x_temp = T.tensor4()

# Test generator in sampling procedure
samplefun = th.function(inputs=[],outputs=gen_dat)
sample_x = []
for k in range(20):
    sample_x.append(samplefun())
sample_x = np.concatenate(sample_x, axis=0)

img_bhwc = np.transpose(sample_x[:100,], (0, 2, 3, 1))
img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
img = plotting.plot_img(img_tile, title= '')
plotting.plt.savefig(args.data_name+"_patches_gen.png")

img_bhwc = np.transpose(trainx[:100,], (0, 2, 3, 1))
img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
img = plotting.plot_img(img_tile, title= '')
plotting.plt.savefig(args.data_name+"_patches_train.png")


evens = [x for x in range(int(test_matched.shape[0])) if x%2 == 0]
odds = [x for x in range(int(test_matched.shape[0])) if x%2 == 1]

disc_layers.append(ll.GlobalPoolLayer(disc_layers[f_low_dim]))
batch_size_test = 100
print('Extracting features from matched data...')
features_matched = extract_features(disc_layers, test_matched, batch_size_test)

print('Extracting features from unmatched data...')
features_unmatched = extract_features(disc_layers, test_unmatched, batch_size_test)

print('Number of features considered in the experiment: ' + str(features_matched[evens].shape[1]))

features_matched[features_matched >= 0.0] = 1
features_matched[features_matched <  0.0] = 0

features_unmatched[features_unmatched >= 0.0] = 1
features_unmatched[features_unmatched < 0.0] = 0

d_matched = hamming_dist(features_matched[evens], features_matched[odds])
d_unmatched = hamming_dist(features_unmatched[evens], features_unmatched[odds])

d_matched = np.diagonal(np.array(d_matched)).tolist()
d_unmatched = np.diagonal(np.array(d_unmatched)).tolist()

curve = roc(d_matched, d_unmatched)

result = fp_at_95(curve)

print('FP at 95 TP value: ' + str(result))

