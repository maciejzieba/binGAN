import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
import nn
import os
import sys

# settings
from backends import get_generator, get_discriminator_brown, get_discriminator_cifar
from data_preprocessing import get_data_patches_training, get_train_data_cifar
from settings import get_settings
from model_utils import load_model
import time


args = get_settings()
# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

if args.dataset_type == 'brown':
    trainx = get_data_patches_training(args.data_name, args.data_dir)
else:
    trainx, _, _, _ = get_train_data_cifar(args.data_dir, args.seed_data)
trainx_unl = trainx.copy()
nr_batches_train = int(trainx.shape[0]/args.batch_size)


# specify generative model
gen_layers = get_generator(args.batch_size, theano_rng)
gen_dat = ll.get_output(gen_layers[-1])


# specify discriminative model
if args.dataset_type == 'brown':
    disc_layers, f_low_dim, f_high_dim = get_discriminator_brown(args.num_features)
else:
    disc_layers, f_low_dim, f_high_dim = get_discriminator_cifar(args.num_features)
disc_params = ll.get_all_params(disc_layers, trainable=True)


# you can use pretrained models
if args.use_pretrained:
    load_model(gen_layers,args.generator_pretrained)
    load_model(disc_layers,args.discriminator_pretrained)


x_temp = T.tensor4()


temp = ll.get_output(gen_layers[-1], deterministic=False, init=True)
temp = ll.get_output(disc_layers[-1], x_temp, deterministic=False, init=True)
init_updates = [u for l in gen_layers+disc_layers for u in getattr(l,'init_updates',[])]
init_param = th.function(inputs=[x_temp], outputs=None, updates=init_updates)


# costs
x_unl_1 = T.tensor4()

output_before_softmax_unl_1 = ll.get_output(disc_layers[-1], x_unl_1, deterministic=False)
output_before_softmax_gen = ll.get_output(disc_layers[-1], gen_dat, deterministic=False)


l_unl = nn.log_sum_exp(output_before_softmax_unl_1)
l_gen = nn.log_sum_exp(output_before_softmax_gen)


loss_unl = -0.5*T.mean(l_unl) + 0.5*T.mean(T.nnet.softplus(l_unl)) + 0.5*T.mean(T.nnet.softplus(l_gen))


def bre_regularizer(f_1, f2):
    x_bin = f_1/(T.abs_(f_1) + args.gamma)
    x_bin_2 = T.sgn(f2)
    M_reg = T.abs_(T.dot(x_bin_2, x_bin_2.T))/x_bin_2.shape[1]
    M_reg = T.exp(-1.0*M_reg/args.beta)
    loss_rme = T.mean(T.square(T.mean(x_bin, axis=0)))
    l_reg = T.sum(M_reg*(T.abs_(T.dot(x_bin,x_bin.T) -
                                T.dot(x_bin,x_bin.T) *
                                (T.eye(x_bin.shape[0])))))/(x_bin.shape[0]*(x_bin.shape[0]-1)*x_bin.shape[1])

    return loss_rme + l_reg


def dmr_regularizer(f_1, f2):
    x_bin = f_1/(T.abs_(f_1) + args.gamma)
    x_bin_2 = T.sgn(f2)
    M_reg = (T.dot(x_bin_2,x_bin_2.T) - T.dot(x_bin_2,x_bin_2.T) *(T.eye(x_bin_2.shape[0])))/x_bin_2.shape[1]
    M_true = (T.dot(x_bin,x_bin.T) - T.dot(x_bin,x_bin.T) *(T.eye(x_bin.shape[0])))/x_bin.shape[1]
    l_reg = T.sum(T.abs_(M_reg - M_true))/(x_bin.shape[0]*(x_bin.shape[0]-1))

    return l_reg

if args.dataset_type=='brown':
    global_pool = ll.GlobalPoolLayer(disc_layers[f_low_dim])
    global_pool_2 = ll.ReshapeLayer(disc_layers[f_high_dim], ([0], -1))
else:
    global_pool = disc_layers[f_low_dim]
    global_pool_2 = ll.GlobalPoolLayer(disc_layers[f_high_dim])

features_1 = ll.get_output(global_pool, x_unl_1, deterministic=False)
features_2 = ll.get_output(global_pool_2, x_unl_1, deterministic=False)
loss_unl = loss_unl + args.lambda_dmr * dmr_regularizer(features_1, features_2) \
           + args.lambda_bre * bre_regularizer(features_1, features_2)

# Theano functions for training the disc net
lr = T.scalar()
disc_params = ll.get_all_params(disc_layers, trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss_unl, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)] # data based initialization
train_batch_disc = th.function(inputs=[x_unl_1,lr], outputs=loss_unl, updates=disc_param_updates)


# Theano functions for training the gen net
output_unl = ll.get_output(disc_layers[-2], x_unl_1, deterministic=False)
output_gen = ll.get_output(disc_layers[-2], gen_dat, deterministic=False)

m1 = T.mean(output_unl,axis=0)
m2 = T.mean(output_gen,axis=0)
loss_gen = T.mean(T.square(m1-m2)) # feature matching loss

gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
train_batch_gen = th.function(inputs=[x_unl_1, lr], outputs=loss_gen, updates=gen_param_updates)

# //////////// perform training //////////////
for epoch in range(args.num_epochs):
    begin = time.time()
    lr = np.cast[th.config.floatX](args.learning_rate * np.minimum(3. - epoch/400., 1.))
    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    trainx_unl_2 = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    if epoch == 0 and not args.use_pretrained:
        print(trainx.shape)
        init_param(trainx[:500]) # data based initialization

    # train
    loss_lab = 0.
    loss_unl = 0.
    train_err = 0.
    l_gen = 0.
    for t in range(nr_batches_train):
        mbatch = trainx_unl[t * args.batch_size:(t + 1) * args.batch_size]
        lu = train_batch_disc(mbatch, lr)
        loss_unl += lu
        lg = train_batch_gen(trainx_unl_2[t*args.batch_size:(t+1)*args.batch_size],lr)
        l_gen += lg

    loss_unl /= nr_batches_train
    train_err /= nr_batches_train
    l_gen /= nr_batches_train
    # test
    test_err = 0.
    # report
    print("Iteration %d, time = %ds, loss_unl = %.4f, l_gen = %.4f" % (epoch, time.time()-begin, loss_unl, l_gen))
    sys.stdout.flush()


    if epoch == 0:
        results = np.array([epoch, time.time()-begin, loss_unl, l_gen]).reshape((1,4))
    else:
        results = np.concatenate((results, np.array([epoch, time.time()-begin, loss_unl, l_gen]).reshape((1,4))),axis=0)

    np.savetxt("training_info_"+args.data_name+"_"+str(args.num_features)+".txt",results,fmt='%.4f',delimiter=',')
    if not os.path.exists(os.path.dirname(args.discriminator_out)):
        os.makedirs(os.path.dirname(args.discriminator_out))
    np.savez(args.discriminator_out, *lasagne.layers.get_all_param_values(disc_layers))
    if not os.path.exists(os.path.dirname(args.generator_out)):
        os.makedirs(os.path.dirname(args.generator_out))
    np.savez(args.generator_out, *lasagne.layers.get_all_param_values(gen_layers))
