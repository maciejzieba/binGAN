import theano.tensor as T
import nn
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn


def get_generator(batch_size, theano_rng, noise_length=100):

    noise_dim = (batch_size, noise_length)
    noise = theano_rng.uniform(size=noise_dim)
    gen_layers = [ll.InputLayer(shape=noise_dim, input_var=noise)]
    gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu), g=None))
    gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (batch_size,512,4,4)))
    gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (batch_size,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 4 -> 8
    gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (batch_size,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 8 -> 16
    gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (batch_size,3,32,32), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1)) # 16 -> 32

    return gen_layers


def get_discriminator_brown(num_feature=256):

    disc_layers = [ll.InputLayer(shape=(None, 3, 32, 32))]
    disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.2))
    disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
    disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
    disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 128, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=num_feature, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=128, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
    disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=2, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))
    #disc_layers.append(ll.ReshapeLayer(disc_layers[-4], ([0], -1)))
    #disc_layers.append(ll.GlobalPoolLayer(disc_layers[-4]))
    disc_layer_features_low_dim = -4
    disc_layer_features_high_dim = -5

    return disc_layers, disc_layer_features_low_dim, disc_layer_features_high_dim



def get_discriminator_cifar(num_feature=24):

    disc_layers = [ll.InputLayer(shape=(None, 3, 32, 32))]
    disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.2))
    disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 96, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
    disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(ll.DropoutLayer(disc_layers[-1], p=0.5))
    disc_layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(disc_layers[-1], 192, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(nn.weight_norm(ll.NINLayer(disc_layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
    disc_layers.append(ll.GlobalPoolLayer(disc_layers[-1]))
    disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=num_feature, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))
    disc_layers.append(nn.weight_norm(ll.DenseLayer(disc_layers[-1], num_units=2, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))
    disc_layer_features_low_dim = -2
    disc_layer_features_high_dim = -4

    return disc_layers, disc_layer_features_low_dim, disc_layer_features_high_dim