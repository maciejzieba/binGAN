import cifar10_data
import numpy as np
import theano.tensor as T
import lasagne.layers as ll
import sys
import theano as th
import os

from six.moves import urllib


def get_train_data_cifar(data_dir, seed_data):
    rng_data = np.random.RandomState(seed_data)
    trainx, trainy = cifar10_data.load(data_dir, subset='train')
    inds = rng_data.permutation(trainx.shape[0])
    trainx = trainx[inds]
    trainy = trainy[inds]
    txs = []
    tys = []
    for j in range(10):
        txs.append(trainx[trainy==j])
        tys.append(trainy[trainy==j])

    txs = np.concatenate(txs, axis=0)
    tys = np.concatenate(tys, axis=0)

    return trainx, trainy, txs, tys


def get_test_data_cifar(data_dir, reduce_to=100, seed_data=1):
    rng_data = np.random.RandomState(seed_data)
    testx, testy = cifar10_data.load(data_dir, subset='test')
    if reduce_to:
        inds = rng_data.permutation(testx.shape[0])
        testx = testx[inds]
        testy = testy[inds]
        testx = testx[0:reduce_to]
        testy = testy[0:reduce_to]
    return testx, testy


def get_feature_extractor(model, x_temp = T.tensor4()):
    features = ll.get_output(model[-1], x_temp, deterministic=True)
    return th.function(inputs=[x_temp], outputs=features)


def extract_features(model, input_data, batch_size=1, x_temp = T.tensor4()):
    nr_batches = int(input_data.shape[0]/batch_size)
    extract = get_feature_extractor(model, x_temp)
    for t in range(nr_batches):
        if t == 0:
            features = extract(input_data[t*batch_size:(t+1)*batch_size])
        else:
            features = np.concatenate((features, extract(input_data[t*batch_size:(t+1)*batch_size])),axis=0)
    return features


def get_feature_extractor_trans(model_D, model_bin, x_temp = T.tensor4()):
    features = ll.get_output(ll.GlobalPoolLayer(model_D), x_temp, deterministic=True)
    features = features / (T.abs_(features) + 0.001)
    features = ll.get_output(model_bin, features, deterministic=False)
    return th.function(inputs=[x_temp], outputs=features)


def extract_features_trans(model_D,model_bin, input_data, batch_size=1, x_temp = T.tensor4()):
    nr_batches = int(input_data.shape[0]/batch_size)
    extract = get_feature_extractor_trans(model_D, model_bin, x_temp)
    for t in range(nr_batches):
        if t == 0:
            features = extract(input_data[t*batch_size:(t+1)*batch_size])
        else:
            features = np.concatenate((features, extract(input_data[t*batch_size:(t+1)*batch_size])),axis=0)
    return features

def maybe_download(data_dir, filename,  url='https://www.ii.pwr.edu.pl/~zieba/datasets/'):
    if not os.path.exists(os.path.join(data_dir, filename)):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(url+filename, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

def get_data_patches_training(dataset_name, data_dir, type="train"):
    file_matched = dataset_name + '_' + type + '_match.npy'
    file_unmatched = dataset_name + '_' + type + '_non-match.npy'
    maybe_download(data_dir, file_matched)
    data_matched = np.load(os.path.join(data_dir, file_matched))
    maybe_download(data_dir, file_unmatched)
    data_non_matched = np.load(os.path.join(data_dir, file_unmatched))
    data = np.concatenate((data_matched, data_non_matched), axis=0).astype(np.float32)
    return data


def get_data_patches_test(dataset_name, data_dir, type="test"):
    file_matched = dataset_name + '_' + type + '_match.npy'
    file_unmatched = dataset_name + '_' + type + '_non-match.npy'
    maybe_download(data_dir, file_matched)
    data_matched = np.load(os.path.join(data_dir, dataset_name + '_'+type+'_match.npy'))
    maybe_download(data_dir, file_unmatched)
    data_non_matched = np.load(os.path.join(data_dir, dataset_name + '_'+type+'_non-match.npy'))
    return data_matched, data_non_matched


