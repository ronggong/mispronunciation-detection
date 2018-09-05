import sys
import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_net.training_scripts.models_TCN import train_TCN_batch
from neural_net.training_scripts.models_RNN import eval_RNN_model
from neural_net.combine_feature_label import combine_feature_label
from neural_net.file_path import *


if __name__ == '__main__':

    cv_prod = "cv"
    batch_size = 1
    input_shape = (batch_size, None, 80)
    patience = 15
    attention = False
    dropout = 0.05
    epoch = 500

    path_model = '/Users/ronggong/PycharmProjects/mispronunciation-detection/neural_net/model/'

    with open(dict_jianzi_positive, "rb") as f:
        feature_jianzi_pos = pickle.load(f)

    with open(dict_jianzi_negative, "rb") as f:
        feature_jianzi_neg = pickle.load(f)

    X_jianzi, y_jianzi = combine_feature_label(dict_positive=feature_jianzi_pos,
                                               dict_negative=feature_jianzi_neg)

    if cv_prod == "cv":
        list_loss = []
        list_acc = []
        skf = StratifiedKFold(n_splits=5)
        for ii, (train_index, val_index) in enumerate(skf.split(X_jianzi, y_jianzi)):

            model_name = 'jianzi_model_tcn_1_stack_3_{}'.format(dropout)
            file_path_model = os.path.join(path_model, model_name + '_' + str(ii) + '.h5')
            file_path_log = os.path.join(path_model, 'log', model_name + '_' + str(ii) + '.csv')

            print("TRAIN:", train_index, "TEST:", val_index)

            X_train, X_test = [X_jianzi[ii] for ii in train_index], [X_jianzi[ii] for ii in val_index]
            y_train, y_test = y_jianzi[train_index], y_jianzi[val_index]

            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1)

            # standarization
            scaler = StandardScaler()
            X_train_conc = np.concatenate(X_train)
            scaler.fit(X_train_conc)

            model = train_TCN_batch(list_feature_fold_train=X_train,
                                    labels_fold_train=y_train,
                                    list_feature_fold_val=X_val,
                                    labels_fold_val=y_val,
                                    batch_size=batch_size,
                                    input_shape=input_shape,
                                    file_path_model=file_path_model,
                                    filename_log=file_path_log,
                                    epoch=epoch,
                                    patience=patience,
                                    scaler=scaler,
                                    dropout=dropout,
                                    summ=True,
                                    verbose=2)

            loss_test = eval_RNN_model(list_feature_test=X_test,
                                       labels_test=y_test,
                                       file_path_model=file_path_model,
                                       attention=attention,
                                       scaler=scaler)

            list_loss.append(loss_test)

        with open(os.path.join(path_model, 'log', 'jianzi_esults_tcn_1_stack_3_{}.txt'.format(dropout)), 'w') as f:
            f.write("loss {}".format(np.mean(list_loss)))

    elif cv_prod == "prod":
        X_train, X_val, y_train, y_val = train_test_split(X_jianzi, y_jianzi, stratify=y_jianzi, test_size=0.1)

        model_name = 'jianzi_model_prod_tcn_{}'.format(dropout)
        file_path_model = os.path.join(path_model, model_name + '.h5')
        file_path_log = os.path.join(path_model, 'log', model_name + '.csv')

        # standarization
        scaler = StandardScaler()
        X_train_conc = np.concatenate(X_train)
        scaler.fit(X_train_conc)

        train_TCN_batch(list_feature_fold_train=X_train,
                        labels_fold_train=y_train,
                        list_feature_fold_val=X_val,
                        labels_fold_val=y_val,
                        batch_size=batch_size,
                        input_shape=input_shape,
                        file_path_model=file_path_model,
                        filename_log=file_path_log,
                        epoch=epoch,
                        patience=patience,
                        scaler=scaler,
                        dropout=dropout,
                        summ=True,
                        verbose=2)
    else:
        raise ValueError("{} is not a valid option.".format(cv_prod))
