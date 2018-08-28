import pickle
import numpy as np
from neural_net.file_path import *


def combine_feature_label(dict_positive, dict_negative):
    """
    Combine positive and negative features and labels into two lists
    :param dict_positive:
    :param dict_negative:
    :return:
    """
    X = []
    y = []
    for key in dict_positive:
        X += dict_positive[key]
        y += [1]*len(dict_positive[key])

    for key in dict_negative:
        X += dict_negative[key]
        y += [0]*len(dict_negative[key])

    return X, np.array(y)


if __name__ == "__main__":
    with open(dict_special_positive, "rb") as f:
        feature_special_pos = pickle.load(f)

    with open(dict_special_negative, "rb") as f:
        feature_special_neg = pickle.load(f)

    with open(dict_jianzi_positive, "rb") as f:
        feature_jianzi_pos = pickle.load(f)

    with open(dict_jianzi_negative, "rb") as f:
        feature_jianzi_neg = pickle.load(f)

    X_special, y_special = combine_feature_label(dict_positive=feature_special_pos,
                                                 dict_negative=feature_special_neg)

    X_jianzi, y_jianzi = combine_feature_label(dict_positive=feature_jianzi_pos,
                                               dict_negative=feature_jianzi_neg)

    print(np.count_nonzero(y_special), len(y_special))
    print(np.count_nonzero(y_jianzi), len(y_jianzi))