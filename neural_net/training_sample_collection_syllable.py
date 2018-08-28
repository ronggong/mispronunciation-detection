import os
import pickle
import json
from neural_net.parameters import *
from neural_net.file_path import *
from neural_net.utils.audio_preprocessing import get_log_mel_madmom
from neural_net.file_path import parse_recordings
from neural_net.utils.textgrid_preprocessing import parse_syllable_line_list


def dump_feature_syllable(recordings, list_normal_special, list_normal_jianzi):

    # feature dictionary
    dic_syllable_special = {}
    dic_syllable_jianzi = {}
    dic_syllable_special_normal = {}
    dic_syllable_jianzi_normal = {}

    for rec in recordings:
        data_path, sub_folder, textgrid_folder, \
        wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
        phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)

        wav_filename = os.path.join(path_root, data_path, wav_folder, sub_folder, filename + ".wav")
        textgrid_filename = os.path.join(path_root, data_path, textgrid_folder, sub_folder, filename + ".textgrid")

        print("Parse textgrid file {}".format(textgrid_filename))

        nested_syllable_list, is_file_exist, is_syllable_found = \
            parse_syllable_line_list(ground_truth_text_grid_file=textgrid_filename,
                                     parent_tier=longsyllable_tier,
                                     child_tier=syllable_tier)

        nested_special_list, is_file_exist, is_special_found = \
            parse_syllable_line_list(ground_truth_text_grid_file=textgrid_filename,
                                     parent_tier=longsyllable_tier,
                                     child_tier=special_tier)

        nested_specialClass_list, is_file_exist, is_specialClass_found = \
            parse_syllable_line_list(ground_truth_text_grid_file=textgrid_filename,
                                     parent_tier=longsyllable_tier,
                                     child_tier=special_class_tier)

        log_mel = get_log_mel_madmom(audio_fn=wav_filename,
                                     fs=fs,
                                     hopsize_t=hopsize_t,
                                     channel=1,
                                     context=False)

        for ii_line in range(len(nested_special_list)):
            line_special_list = nested_special_list[ii_line]
            if line_special_list[0][2] != "1":
                line_syllable_list = nested_syllable_list[ii_line]
                line_specialClass_list = nested_specialClass_list[ii_line]

                for ii_syl in range(len(line_specialClass_list[1])):
                    special_class = line_specialClass_list[1][ii_syl][2]
                    try:
                        syllable = line_syllable_list[1][ii_syl][2]
                    except IndexError:
                        raise IndexError(rec, ii_line)

                    if special_class == "1" or special_class == "2":
                        label_special = line_special_list[1][ii_syl][2]
                        onset = line_special_list[1][ii_syl][0]
                        offset = line_special_list[1][ii_syl][1]
                        sf = int(round(onset * fs / float(hopsize)))  # starting frame
                        ef = int(round(offset * fs / float(hopsize)))  # ending frame
                        log_mel_syllable = log_mel[sf:ef, :]

                        if len(log_mel_syllable):
                            if special_class == "1":  # shangkou
                                num_special += 1
                                if label_special in dic_syllable_special:
                                    dic_syllable_special[label_special].append(log_mel_syllable)
                                else:
                                    dic_syllable_special[label_special] = [log_mel_syllable]
                            if special_class == "2":  # jiantuan
                                num_jianzi += 1
                                if label_special in dic_syllable_jianzi:
                                    dic_syllable_jianzi[label_special].append(log_mel_syllable)
                                else:
                                    dic_syllable_jianzi[label_special] = [log_mel_syllable]

                    elif not special_class.isdigit():
                        onset = line_syllable_list[1][ii_syl][0]
                        offset = line_syllable_list[1][ii_syl][1]
                        sf = int(round(onset * fs / float(hopsize)))  # starting frame
                        ef = int(round(offset * fs / float(hopsize)))  # ending frame
                        log_mel_syllable = log_mel[sf:ef, :]

                        if len(log_mel_syllable):
                            if syllable in list_normal_special:
                                if syllable in dic_syllable_special_normal:
                                    dic_syllable_special_normal[syllable].append(log_mel_syllable)
                                else:
                                    dic_syllable_special_normal[syllable] = [log_mel_syllable]
                            if syllable in list_normal_jianzi:
                                if syllable in dic_syllable_jianzi_normal:
                                    dic_syllable_jianzi_normal[syllable].append(log_mel_syllable)
                                else:
                                    dic_syllable_jianzi_normal[syllable] = [log_mel_syllable]
                    else:
                        pass

    return dic_syllable_special, dic_syllable_jianzi, dic_syllable_special_normal, dic_syllable_jianzi_normal


if __name__ == "__main__":

    with open(filename_normal_special, "r") as f:
        list_normal_special = json.load(f)
    with open(filename_normal_jianzi, "r") as f:
        list_normal_jianzi = json.load(f)

    dic_syllable_special, dic_syllable_jianzi, dic_syllable_special_normal, dic_syllable_jianzi_normal = \
        dump_feature_syllable(recordings=recordings_train,
                              list_normal_special=list_normal_special,
                              list_normal_jianzi=list_normal_jianzi)

    with open(dict_special_positive, "wb") as f:
        pickle.dump(dic_syllable_special, f)

    with open(dict_special_negative, "wb") as f:
        pickle.dump(dic_syllable_special_normal, f)

    with open(dict_jianzi_positive, "wb") as f:
        pickle.dump(dic_syllable_jianzi, f)

    with open(dict_jianzi_negative, "wb") as f:
        pickle.dump(dic_syllable_jianzi_normal, f)
