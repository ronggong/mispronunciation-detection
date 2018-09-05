from neural_net.keras_tcn.tcn import tcn
from neural_net.training_scripts.generator import generator_batch1
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint


def train_TCN_batch(list_feature_fold_train,
                    labels_fold_train,
                    list_feature_fold_val,
                    labels_fold_val,
                    batch_size,
                    input_shape,
                    file_path_model,
                    filename_log,
                    epoch,
                    patience,
                    scaler,
                    dropout,
                    summ=False,
                    verbose=2):

    model, param_str = tcn.dilated_tcn(output_slice_index='last',  # try 'first'.
                                       num_feat=input_shape[-1],
                                       num_classes=2,
                                       nb_filters=16,
                                       kernel_size=3,
                                       dilatations=[0, 1, 3, 5],
                                       nb_stacks=1,
                                       max_len=None,
                                       dropout=dropout,
                                       activation='norm_relu',
                                       use_skip_connections=False,
                                       return_param_str=True)

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