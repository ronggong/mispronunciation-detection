from keras.models import Input
from keras.models import Model
from keras.models import load_model
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Reshape
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib

import os
import sys
import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from neural_net.training_scripts.attention import Attention
from neural_net.training_scripts.generator import generator_batch1


def conv_module(conv, input_shape, input):
    if conv:
        x = Reshape((-1, input_shape[2]) + (1,))(input)
        x = Conv2D(filters=8, kernel_size=(1, 3), activation="relu")(x)
        x = MaxPooling2D(pool_size=(1, 3))(x)

        x = Conv2D(filters=16, kernel_size=(1, 3), activation="relu")(x)
        x = MaxPooling2D(pool_size=(1, 3))(x)
        shape = K.int_shape(x)
        x = Reshape((-1, shape[2] * shape[3]))(x)
    else:
        x = input
    return x


def embedding_RNN_1_lstm(input_shape, conv=False, dropout=False, att=False):

    device = device_lib.list_local_devices()[0].device_type

    input = Input(batch_shape=input_shape)

    x = conv_module(conv, input_shape, input)

    if att:
        return_sequence = True
    else:
        return_sequence = False

    if device == 'CPU':
        if dropout:
            x = Bidirectional(LSTM(units=8, return_sequences=return_sequence, dropout=dropout))(x)
            x = Dropout(dropout)(x)
        else:
            x = Bidirectional(LSTM(units=8, return_sequences=return_sequence))(x)
    else:
        x = Bidirectional(CuDNNLSTM(units=8, return_sequences=return_sequence))(x)

    if att:
        print(K.shape(x))
        x, attention = Attention(return_attention=True)(x)
    else:
        attention = None

    return x, input, attention


def train_RNN_batch(list_feature_fold_train,
                    labels_fold_train,
                    list_feature_fold_val,
                    labels_fold_val,
                    batch_size,
                    input_shape,
                    output_shape,
                    file_path_model,
                    filename_log,
                    epoch,
                    patience,
                    scaler,
                    attention,
                    conv,
                    dropout,
                    summ=False,
                    verbose=2):

    x, input, _ = embedding_RNN_1_lstm(input_shape=input_shape,
                                       conv=conv,
                                       dropout=dropout,
                                       att=attention)

    outputs = Dense(output_shape, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    if summ:
        model.summary()

    callbacks = [ModelCheckpoint(file_path_model, monitor='val_loss', verbose=0, save_best_only=True),
                 EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
                 CSVLogger(filename=filename_log, separator=';')]

    print("start training with validation...")

    generator_train = generator_batch1(list_feature=list_feature_fold_train,
                                       labels=labels_fold_train,
                                       scaler=scaler)

    generator_val = generator_batch1(list_feature=list_feature_fold_val,
                                     labels=labels_fold_val,
                                     scaler=scaler)

    model.fit_generator(generator=generator_train,
                        steps_per_epoch=len(list_feature_fold_train)/batch_size,
                        validation_data=generator_val,
                        validation_steps=len(list_feature_fold_val)/batch_size,
                        callbacks=callbacks,
                        epochs=epoch,
                        verbose=verbose)

    return model


def eval_RNN_model(list_feature_test,
                   labels_test,
                   file_path_model,
                   attention,
                   scaler):
    if attention:
        model = load_model(filepath=file_path_model,
                           custom_objects={'Attention': Attention(return_attention=True)})
    else:
        model = load_model(file_path_model)

    list_y_pred = np.zeros((len(labels_test, )))
    for ii in range(len(list_feature_test)):
        fea = list_feature_test[ii]
        fea = scaler.transform(fea)
        fea = np.expand_dims(fea, axis=0)
        y_pred = model.predict_on_batch(fea)
        list_y_pred[ii] = y_pred[0][0]

    loss_test = log_loss(y_true=labels_test, y_pred=list_y_pred)

    return loss_test