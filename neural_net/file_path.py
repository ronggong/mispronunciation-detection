import os
from neural_net.utils.csv_preprocessing import open_csv_recordings

dir_path = os.path.dirname(os.path.realpath(__file__))

path_root = '/Users/ronggong/Documents_using/MTG_document/Jingju_arias/'

path_nacta = 'jingju_a_cappella_singing_dataset'
path_nacta2017 = 'jingju_a_cappella_singing_dataset_extended_nacta2017'
path_primary = 'primary_school_recording'

recordings_train = open_csv_recordings(os.path.join(dir_path, "data/mispronunciation_filelist_train.csv"))
recordings_test = open_csv_recordings(os.path.join(dir_path, "data/mispronunciation_filelist_test.csv"))

filename_normal_special = os.path.join(dir_path, "data/normal_special.json")
filename_normal_jianzi = os.path.join(dir_path, "data/normal_jianzi.json")

dict_special_positive = os.path.join(dir_path, "data/special_positive.pkl")
dict_special_negative = os.path.join(dir_path, "data/special_negative.pkl")
dict_jianzi_positive = os.path.join(dir_path, "data/jianzi_positive.pkl")
dict_jianzi_negative = os.path.join(dir_path, "data/jianzi_negative.pkl")

joint_cnn_model_path = os.path.join(dir_path, 'model', 'segmentation')

filename_special_model = os.path.join(dir_path, "model", "special_model_prod_True_True_0.5.h5")
filename_jianzi_model = os.path.join(dir_path, "model", "jianzi_model_prod_True_True_0.5.h5")
# filename_jianzi_model = os.path.join(dir_path, "model", "jianzi_model_prod_feedforward_True_0.5.h5")

# filename_special_model = os.path.join(dir_path, "model", "special_model_prod_tcn_0.05.h5")
# filename_jianzi_model = os.path.join(dir_path, "model", "jianzi_model_prod_tcn_0.05.h5")

filename_result_decoded_mispronunciaiton = os.path.join(dir_path, "results", "text_decoded_special_True_True_0.5")
# filename_result_decoded_mispronunciaiton = os.path.join(dir_path, "results", "text_decoded_special_feedforward_True_0.5")

path_figs_jianzi = "/Users/ronggong/PycharmProjects/mispronunciation-detection/neural_net/figs/jianzi"


def getRecordings(wav_path):
    recordings = []
    for root, subFolders, files in os.walk(wav_path):
        for f in files:
            file_prefix, file_extension = os.path.splitext(f)
            if file_prefix != '.DS_Store':
                recordings.append(file_prefix)

    return recordings


def parse_recordings(rec):
    if rec[0] == "part1":
        data_path = path_nacta
        sub_folder = rec[2]
        textgrid_folder = "textgrid"
        wav_folder = "wav_left"
        syllable_tier = "dian"
        if rec[3][:2] == 'da':
            roletype = 'Dan'
        elif rec[3][:2] == 'ls':
            roletype = 'Laosheng'
        else:
            raise ValueError("Not exist a role-type {} for file {}".format(rec[3][:2], rec))
    elif rec[0] == "part2":
        data_path = path_nacta2017
        sub_folder = rec[2]
        textgrid_folder = "textgridDetails"
        wav_folder = "wav"
        syllable_tier = "dianSilence"
        if rec[3][:2] == 'da':
            roletype = 'Dan'
        elif rec[3][:2] == 'ls':
            roletype = 'Laosheng'
        else:
            raise ValueError("Not exist a role-type {} for file {}".format(rec[3][:2], rec))
    else:
        data_path = path_primary
        sub_folder = rec[1] + "/" + rec[2]
        textgrid_folder = "textgrid"
        wav_folder = "wav_left"
        syllable_tier = "dianSilence"
        if rec[2][:2] == 'da':
            roletype = 'Dan'
        elif rec[2][:2] == 'ls':
            roletype = 'Laosheng'
        else:
            raise ValueError("Not exist a role-type {} for file {}".format(rec[2][:2], rec))

    filename = rec[3]
    line_tier = "line"
    longsyllable_tier = "longsyllable"
    phoneme_tier = "details"
    special_tier = "special"
    special_class_tier = "specialClass"

    return data_path, sub_folder, textgrid_folder, \
           wav_folder, filename, line_tier, \
           longsyllable_tier, syllable_tier, phoneme_tier, \
           special_tier, special_class_tier, roletype