import os
import json
from kaldi_alignment.srcPy.filePath import *
from kaldi_alignment.srcPy.textgridParser import syllableTextgridExtraction


def collectLexicon(path_textgrid, recording, tier0, tier1, lexicon):
    nestedSyllableLists, numLines, numSyllables = syllableTextgridExtraction(path_textgrid,
                                                                             recording,
                                                                             tier0,
                                                                             tier1)
    for syl in nestedSyllableLists:
        lexicon_unit = syl[0][2].strip().upper()+' '
        for ii_pho, pho in enumerate(syl[1]):
            # if pho[2] == '?':
            # 	continue
            if not len(pho[2]) and ii_pho >= len(syl[1])-1:
                lexicon_unit = lexicon_unit[:-1]  # remove the last space
                break
            if len(pho[2]):
                lexicon_unit += pho[2]
            else:
                lexicon_unit += 'sil_phone'
            if ii_pho != len(syl[1])-1:
                lexicon_unit += ' '  # add a space in the end of the char
        lexicon.append(lexicon_unit)
    return lexicon


def collect_lexicon_syllable_special(path_textgrid, recording, syllable_tier, special_tier, phoneme_tier, lexicon):
    nestedSyllableLists, numLines, numSyllables = syllableTextgridExtraction(path_textgrid,
                                                                             recording,
                                                                             syllable_tier,
                                                                             phoneme_tier)

    nestedSpecialLists, numLines, numSpecial = syllableTextgridExtraction(path_textgrid,
                                                                          recording,
                                                                          special_tier,
                                                                          phoneme_tier)
    for ii_syl, syl in enumerate(nestedSyllableLists):
        lexicon_unit = syl[0][2].strip().upper() + ' ' + nestedSpecialLists[ii_syl][0][2].strip().upper() + ' '
        for ii_pho, pho in enumerate(syl[1]):
            # if pho[2] == '?':
            # 	continue
            if not len(pho[2]) and ii_pho >= len(syl[1])-1:
                lexicon_unit = lexicon_unit[:-1]  # remove the last space
                break
            if len(pho[2]):
                lexicon_unit += pho[2]
            else:
                lexicon_unit += 'sil_phone'
            if ii_pho != len(syl[1])-1:
                lexicon_unit += ' '  # add a space in the end of the char
        lexicon.append(lexicon_unit)
    return lexicon


def organizeRepetition(lexicon, repetition=False):
    """
    give the syllable repetition different name
    """
    names_syllable 		= {}
    lexicon_organized 	= []
    dict_lexicon_organized = {}
    for l in lexicon:
        syls = l.split()

        if repetition:
            if syls[0] not in names_syllable.keys():
                names_syllable[syls[0]] = 0
            else:
                names_syllable[syls[0]] += 1

            syls[0] = syls[0]+str(names_syllable[syls[0]])

        lexicon_unit = ' '.join(syls)

        # remove repetition
        if lexicon_unit not in lexicon_organized:
            lexicon_organized.append(lexicon_unit)
            dict_lexicon_organized[syls[0]] = syls[1:]
    return lexicon_organized, dict_lexicon_organized


def organize_repetition_syllable_special(lexicon, repetition=False):
    """
    give the syllable repetition different name, lexicon contains syllable and special
    """
    names_syllable 		= {}
    lexicon_organized 	= []
    dict_lexicon_organized = {}

    for l in lexicon:
        print(l)
        syls = l[0].split()

        if repetition:
            if syls[0] not in names_syllable.keys():
                names_syllable[syls[0]] = 0
            else:
                names_syllable[syls[0]] += 1

            syls[0] = syls[0]+str(names_syllable[syls[0]])

        lexicon_unit = [' '.join(syls), l[1:]]

        # remove repetition
        if lexicon_unit[0] not in lexicon_organized:
            lexicon_organized.append(lexicon_unit[0])
            dict_lexicon_organized[syls[0]] = [syls[1:], l[1:]]
    return lexicon_organized, dict_lexicon_organized


def writeLexicon(path_lang, lexicon, repetition=False):
    if repetition:
        filename_lexicon = 'lexicon.txt'
    else:
        filename_lexicon = 'lexicon_no_rep.txt'

    with open(os.path.join(path_lang, filename_lexicon), "w") as f:
        f.write('SIL sil\n')
        # f.write('SIL_PHONE sil_phone\n')
        # f.write('SILDAN silDan\n')
        f.write('<SPOKEN_NOISE> sil\n')

        for l in lexicon:
            f.write(l)
            f.write('\n')


if __name__ == '__main__':

    # train: organize lexicon with repetition,
    # test: organize lexicon without repetition.
    train_test = 'test'

    lexicon = []

    if train_test == 'train':

        for rec in recordings_train+recordings_test:
            data_path, sub_folder, textgrid_folder, \
            wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
            phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)

            lexicon = collectLexicon(path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
                                     recording=filename,
                                     tier0=special_tier,
                                     tier1=phoneme_tier,
                                     lexicon=lexicon)

        lexicon = list(set(lexicon))

        lexicon_organized, dict_lexicon_organized = organizeRepetition(lexicon, repetition=True)

        writeLexicon(path_lang, lexicon_organized, repetition=True)

        with open(os.path.join(path_lang, "dict_lexicon_repetition.json"), "w") as write_file:
            json.dump(dict_lexicon_organized, write_file)
    else:
        lexicon_special = []
        lexicon_syllable_special = []

        for rec in recordings_train+recordings_test:
            data_path, sub_folder, textgrid_folder, \
            wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
            phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)

            lexicon = collectLexicon(path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
                                     recording=filename,
                                     tier0=syllable_tier,
                                     tier1=phoneme_tier,
                                     lexicon=lexicon)

            lexicon_special = collectLexicon(
                path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
                recording=filename,
                tier0=special_tier,
                tier1=phoneme_tier,
                lexicon=lexicon_special)

            lexicon_syllable_special = collect_lexicon_syllable_special(path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
                                                                        recording=filename,
                                                                        syllable_tier=syllable_tier,
                                                                        special_tier=special_tier,
                                                                        phoneme_tier=phoneme_tier,
                                                                        lexicon=lexicon_syllable_special)

        lexicon = list(set(lexicon))

        lexicon_special = list(set(lexicon_special))

        lexicon_syllable_special = list(set(lexicon_syllable_special))

        # get a list ['SYL phn0 phn1 phn2', 'SPECIAL']
        lexicon_remove_rep = []
        for pron_special in lexicon_special:
            lexicon_unit = [pron_special]
            for word_entry in lexicon_syllable_special:
                syl = word_entry.split(' ')[0]
                if pron_special == ' '.join(word_entry.split(' ')[1:]):
                    lexicon_unit.append(syl)
            lexicon_remove_rep.append(lexicon_unit)

        lexicon_organized, dict_lexicon_organized = organizeRepetition(lexicon, repetition=False)

        # dict_lexicon_organized_syllable_special, {SPECIAL: [[phn0 phn1 phn2], [SYL0 SYL1]]}
        lexicon_organized_syllable_special, dict_lexicon_organized_syllable_special = \
            organize_repetition_syllable_special(lexicon_remove_rep, repetition=True)

        with open(os.path.join(path_lang, "dict_lexicon_repetition_syllable_special.json"), "w") as write_file:
            json.dump(dict_lexicon_organized_syllable_special, write_file)