"""
Functions related to normal pronunciation manipulation
"""
import os
import json
from neural_net.file_path import path_root
from neural_net.file_path import recordings_train
from neural_net.file_path import parse_recordings
from neural_net.utils.textgrid_preprocessing import parse_syllable_line_list


if __name__ == "__main__":

    list_normal_special = []  # the normal counterpart of the special pronunciation
    list_normal_jianzi = []  # the normal counterpart of jianzi

    for rec in recordings_train:
        data_path, sub_folder, textgrid_folder, \
        wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
        phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)

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

        nested_phoneme_list, is_file_exist, is_phoneme_found = \
            parse_syllable_line_list(ground_truth_text_grid_file=textgrid_filename,
                                     parent_tier=longsyllable_tier,
                                     child_tier=phoneme_tier)

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

                    if special_class == "1":  # shangkou
                        shangkou = line_special_list[1][ii_syl][2]
                        list_normal_special.append(syllable)
                        # print("shangkou", syllable, shangkou, rec, ii_line)
                    if special_class == "2":  # jiantuan
                        jiantuan = line_special_list[1][ii_syl][2]
                        list_normal_jianzi.append(syllable)

    list_normal_special = list(set(list_normal_special))
    list_normal_jianzi = list(set(list_normal_jianzi))

    with open("./data/normal_special.json", "w") as f:
        json.dump(list_normal_special, f)

    with open("./data/normal_jianzi.json", "w") as f:
        json.dump(list_normal_jianzi, f)