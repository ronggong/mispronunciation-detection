import os
from os.path import join
from kaldi_alignment.srcPy.textgridParser import syllableTextgridExtraction
from kaldi_alignment.srcPy.csv_prepossessing import open_csv_recordings

path_root = '/Users/ronggong/Documents_using/MTG_document/Jingju_arias/'

path_nacta = 'jingju_a_cappella_singing_dataset'
path_nacta2017 = 'jingju_a_cappella_singing_dataset_extended_nacta2017'
path_primary = 'primary_school_recording'

dataset_laosheng = 'qmLonUpf/laosheng'
dataset_ss = 'sourceSeparation'

path_data_train = '../data/train'
path_data_dev = '../data/dev'
path_data_test = '../data/test'
path_data_LM = '../data/LM'

path_lang = '../data/dict'

recordings_train = open_csv_recordings("mispronunciation_filelist_train.csv")
recordings_test = open_csv_recordings("mispronunciation_filelist_test.csv")


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


if __name__ == '__main__':
    for fn in getRecordings(os.path.join(path_root, path_nacta, 'textgrid', 'laosheng')):
        print('[\'laosheng\', \''+fn+'\'],')