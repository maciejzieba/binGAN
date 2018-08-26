import numpy as np


def hamming_dist(test_features,train_features):
    n_bits = test_features.shape[1]
    Y = n_bits - np.dot(test_features,np.transpose(train_features)) \
        - np.dot(test_features-1,np.transpose(train_features -1))

    return Y


def mean_average_precison(testy,trainy,Y,hamming=True, reduce_to=1000):
    mAP = 0.0;
    N_test = np.shape(testy)[0]
    for k in range(N_test):
        y_true = (testy[k] == trainy)
        y_scores = Y[k,:]
        if reduce_to:
            y_true = y_true[0:reduce_to]
            y_scores = y_scores[0:reduce_to]
        if(hamming):
            ap = average_precision_score_hamming(y_true, y_scores)
        else:
            ap = average_precision_score(y_true, y_scores)
        if not np.isnan(ap):
            mAP = mAP + ap

    return mAP/float(N_test)


def average_precision_score(y_true, y_dist):
    ind = np.argsort(y_dist)
    y_true = y_true[ind]
    N_pos = np.sum(y_true)
    ap = np.cumsum(y_true)*y_true
    ap = ap/np.arange(1,np.shape(y_true)[0]+1).astype(float)
    ap = 1/float(N_pos)*np.sum(ap)
    return ap


def average_precision_score_hamming(y_true, y_dist):
    y_dist = y_dist.astype(int)
    N_pos = float(np.sum(y_true))
    ap = 0.0
    for k in range(0,np.max(y_dist)+1):
        if np.sum(y_true[y_dist==k])>0:
            current_points = y_true[y_dist<=k]
            TP = np.sum(current_points)
            prec = float(TP)/float(current_points.shape[0])
            recall_temp = float(np.sum(y_true[y_dist==k]))/N_pos
            ap = ap + prec*recall_temp
    return ap