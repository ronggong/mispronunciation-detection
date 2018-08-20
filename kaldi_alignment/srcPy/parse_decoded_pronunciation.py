"""
write the decoded text for test set
"""
import json
from kaldi_alignment.srcPy.filePath import *


def open_decoded_pronunciation(filename):
    utt = []
    with open(filename) as file:
        for row in file.readlines():
            utt.append(row.replace('<s>\t', '').replace('\t</s>\n', ''))
    return utt


def parse_lexicon_to_list(lexicon):
    list_lexicon = []
    with open(lexicon) as file:
        for row in file.readlines():
            row = row.replace('\n', '')
            list_lexicon.append([row.split(' ')[0], row.split(' ')[1:]])
    return list_lexicon


def lexicon_finder(dict_lexicon_organized, pho_list):
    """
    find the corresponding pho_list in lexicon organized
    """
    for syl_organized, dict_pho_list in dict_lexicon_organized.items():
        if pho_list == dict_pho_list:
            return syl_organized

    raise ValueError("Not found word entry for {}".format(pho_list))


if __name__ == "__main__":

    path_test_ali = "/Users/ronggong/PycharmProjects/mispronunciation-detection/kaldi_alignment/exp/mono_test_ali/"
    path_lang_test = "/Users/ronggong/PycharmProjects/mispronunciation-detection/kaldi_alignment/data/dict_test"
    filename_decoded_pronunciation = os.path.join(path_test_ali, "pron_perutt_nowb.txt")

    list_lexicon = parse_lexicon_to_list(os.path.join(path_lang_test, 'lexicon.txt'))

    with open(os.path.join(path_lang, "dict_lexicon_repetition_syllable_special.json"), "r") as read_file:
        dict_lexicon = json.load(read_file)

    utts = open_decoded_pronunciation(filename_decoded_pronunciation)

    with open(os.path.join(path_test_ali, 'text_decoded'), "w") as f:
        for utt in utts:
            utt_list = utt.split('\t')
            utt_organized = [utt_list[0]]

            for pho_list in utt_list[1:]:
                if pho_list != 'SIL sil':
                    # find all the pronunciations for the syl in repetitive lexicon
                    syl = pho_list.split(' ')[0]
                    pron_decoded = pho_list.split(' ')[1:]

                    # gather all the special pronunciation for the syl
                    list_syllable_special_unit = []
                    for special, pron_syl in dict_lexicon.items():
                        if syl in pron_syl[1]:
                            list_syllable_special_unit.append([special, pron_syl[0]])

                    # match the special pronunciation
                    for special, pron in list_syllable_special_unit:
                        if pron_decoded == pron:
                            utt_organized.append(''.join([i for i in special if not i.isdigit()]))
                            break
            f.write(' '.join(utt_organized)+'\n')
