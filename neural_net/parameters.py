fs = 44100
framesize_t = 0.025  # in second
hopsize_t = 0.010

framesize = int(round(framesize_t * fs))
hopsize = int(round(hopsize_t * fs))

highFrequencyBound = fs/2 if fs/2 < 11000 else 11000

varin = {}
# parameters of viterbi
varin['delta_mode'] = 'proportion'
varin['delta'] = 0.35


def config_select(config):
    if config[0] == 1 and config[1] == 0:
        model_name = 'single_lstm'
    elif config[0] == 1 and config[1] == 1:
        model_name = 'single_lstm_single_dense'
    elif config[0] == 2 and config[1] == 0:
        model_name = 'two_lstm'
    elif config[0] == 2 and config[1] == 1:
        model_name = 'two_lstm_single_dense'
    elif config[0] == 2 and config[1] == 2:
        model_name = 'two_lstm_two_dense'
    elif config[0] == 3 and config[1] == 0:
        model_name = 'three_lstm'
    elif config[0] == 3 and config[1] == 1:
        model_name = 'three_lstm_single_dense'
    elif config[0] == 3 and config[1] == 2:
        model_name = 'three_lstm_two_dense'
    elif config[0] == 3 and config[1] == 3:
        model_name = 'three_lstm_three_dense'
    else:
        raise ValueError

    return model_name