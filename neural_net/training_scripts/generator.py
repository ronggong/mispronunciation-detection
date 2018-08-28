import numpy as np


def generator_batch1(list_feature, labels, scaler, shuffle=True):
    """data generator"""
    ii = 0
    while True:
        if scaler:
            fea = scaler.transform(list_feature[ii])
        else:
            fea = list_feature[ii]

        fea = np.expand_dims(fea, axis=0)
        lab = np.expand_dims(labels[ii], axis=0)

        yield fea, lab

        ii += 1

        if ii >= len(list_feature):
            ii = 0
            if shuffle:
                p = np.random.permutation(len(list_feature))
                list_feature = [list_feature[ii_p] for ii_p in p]
                labels = labels[p]  # labels is a numpy array