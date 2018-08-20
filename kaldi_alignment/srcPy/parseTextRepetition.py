# -*- coding: utf-8 -*-
"""
Consider syllable repetition,
if word appeared receptively, treat them as different word
"""

import os
import json
from kaldi_alignment.srcPy.filePath import *
from kaldi_alignment.srcPy.textgridParser import syllableTextgridExtraction
from kaldi_alignment.srcPy.parseLang import collectLexicon
from kaldi_alignment.srcPy.parseLang import organizeRepetition


def lexiconFinder(dict_lexicon_organized, syl, nestedPhonemeLists):
	"""
	find the corresponding syl in lexicon organized by nestedPhonemeLists
	"""
	# print nestedPhonemeLists
	for jj_syl_pho_list, syl_pho_list in enumerate(nestedPhonemeLists):
		# print('before',syl,syl_pho_list[0])
		if syl == syl_pho_list[0]:
			# print('after',syl,syl_pho_list[0])
			pho_list = []
			for ii_pho, pho in enumerate(syl_pho_list[1]):
				if ii_pho < len(syl_pho_list[1]) - 1 or len(pho[2]):
					if len(pho[2]):
						pho_list.append(pho[2])
					else:
						pho_list.append('sil_phone')
			nestedPhonemeLists.pop(jj_syl_pho_list)
			break

	for syl_organized, dict_pho_list in dict_lexicon_organized.items():
		if pho_list == dict_pho_list and syl[2].strip().upper() == ''.join([i for i in syl_organized if not i.isdigit()]):
			return syl_organized, nestedPhonemeLists

	raise ValueError("Not found word entry for {}, {}".format(pho_list, syl[2]))


def writeSegments(full_path_textgrid, data_path, sub_folder, rl, tier0, tier1, roletype):

	nestedSyllableLists, numLines, numSyllables = syllableTextgridExtraction(full_path_textgrid, rl, tier0, tier1)

	ii = 0
	for line in nestedSyllableLists:

		if not len(line[1]):
			continue

		# add 0s to sentence number
		zero_adding = ''
		for jj in range(3 - len(str(ii))):
			zero_adding += '0'

		f.write(roletype + '_' + data_path + '_' + sub_folder + '_' + rl + '_' + zero_adding + str(ii) + ' ' +
				roletype + '_' + data_path + '_' + sub_folder + '_' + rl + ' ' +
				"%.2f" % round(line[0][0], 2) + ' ' + "%.2f" % round(line[0][1], 2))
		f.write('\n')

		ii += 1


def writeText(full_path_textgrid,
			  data_path,
			  sub_folder,
			  rl,
			  line_tier,
			  syllable_tier,
			  phoneme_tier,
			  roletype,
			  dict_lexicon):

	nestedSyllableLists, numLines, numSyllables = \
		syllableTextgridExtraction(full_path_textgrid, rl, line_tier, syllable_tier)

	if syllable_tier == "specialTeacher":
		nestedSyllableClassLists, numLines, numSyllables = \
			syllableTextgridExtraction(full_path_textgrid, rl, line_tier, "specialClassTeacher")

	if dict_lexicon:
		nestedPhonemeLists, numSyllables, numPhonemes = \
			syllableTextgridExtraction(full_path_textgrid, rl, syllable_tier, phoneme_tier)

	ii = 0
	for ii_line, line in enumerate(nestedSyllableLists):

		if not len(line[1]):
			continue

		# add 0s to sentence number
		zero_adding = ''
		for jj in range(3 - len(str(ii))):
			zero_adding += '0'

		f.write(roletype + '_' + data_path + '_' + sub_folder + '_' + rl + '_' + zero_adding + str(ii) + ' ')

		for ii_syl, syl in enumerate(line[1]):
			if len(syl[2]):
				if dict_lexicon:
					syl_organized, nestedPhonemeLists = lexiconFinder(dict_lexicon, syl, nestedPhonemeLists)
					f.write(syl_organized)
				else:
					if syllable_tier == "specialTeacher":
						# print(nestedSyllableClassLists[ii_line][1][ii_syl])
						special_class_teacher = nestedSyllableClassLists[ii_line][1][ii_syl][2]
						if special_class_teacher == '1' or special_class_teacher == '2':
							f.write(syl[2].strip().upper() + ' ' + special_class_teacher)
						else:
							f.write(syl[2].strip().upper() + ' ' + '0')
					else:
						f.write(syl[2].strip().upper())
			elif dict_lexicon:
				f.write('SIL')
			else:
				pass
			if len(syl[2]) or dict_lexicon:
				if ii_syl != len(line[1]) - 1:
					f.write(' ')
		f.write('\n')

		ii += 1


def writeWavScp(roletype, data_path, sub_folder, full_path_wav, rl):
	f.write(roletype + '_' + data_path + '_' + sub_folder + '_' + rl + ' ' + full_path_wav)
	f.write('\n')


def writeUtt2spk(full_path_textgrid, data_path, sub_folder, rl, line_tier, syllable_tier, roletype):

	nestedSyllableLists, numLines, numSyllables = syllableTextgridExtraction(full_path_textgrid,
																			 rl,
																			 line_tier,
																			 syllable_tier)
	ii = 0
	for line in nestedSyllableLists:
		if not len(line[1]):
			continue

		# add 0s to sentence number
		zero_adding = ''
		for jj in range(3 - len(str(ii))):
			zero_adding += '0'

		f.write(roletype + '_' + data_path + '_' + sub_folder + '_' + rl + '_' + zero_adding + str(ii) + ' '
				+ roletype)
		f.write('\n')

		ii += 1


def writeSpk2utt(full_path_textgrid,
				 data_path,
				 sub_folder,
				 rl,
				 line_tier,
				 syllable_tier,
				 roletype,
				 utt_list):

	nestedSyllableLists, numLines, numSyllables = \
		syllableTextgridExtraction(full_path_textgrid,
								   rl,
								   line_tier,
								   syllable_tier)
	ii = 0
	for ii_line, line in enumerate(nestedSyllableLists):
		if not len(line[1]):
			continue

		# add 0s to sentence number
		zero_adding = ''
		for jj in range(3 - len(str(ii))):
			zero_adding += '0'

		utt_list.append(roletype + '_' + data_path + '_' + sub_folder + '_' + rl + '_' + zero_adding + str(ii))

		if ii_line != len(nestedSyllableLists):
			utt_list.append(' ')
		ii += 1


def writePhone(full_path_textgrid, data_path, sub_folder, rl, line_tier, phoneme_tier, roletype):

	ii = 0
	nestedSyllableLists, numLines, numSyllables = syllableTextgridExtraction(full_path_textgrid,
																			 rl,
																			 line_tier,
																			 phoneme_tier)
	for line in nestedSyllableLists:
		if not len(line[1]):
			continue

		# add 0s to sentence number
		zero_adding = ''
		for jj in range(3 - len(str(ii))):
			zero_adding += '0'

		f.write(roletype + '_' + data_path + '_' + sub_folder + '_' + rl + '_' + zero_adding + str(ii) + ' ')

		lexicon_unit = ''
		for ii_syl, syl in enumerate(line[1]):
			if not len(syl[2]) and ii_syl >= len(line[1]) - 1:
				lexicon_unit = lexicon_unit[:-1]  # remove the last space
				break
			if len(syl[2]):
				lexicon_unit += syl[2].strip()
			else:
				lexicon_unit += 'sil_phone'
			if ii_syl != len(line[1]) - 1:
				lexicon_unit += ' '

		f.write(''.join(lexicon_unit))
		f.write('\n')

		ii += 1


if __name__ == '__main__':

	# # write segments
	# with open(os.path.join(path_data_train, 'segments'), "w") as f:
	# 	for rec in recordings_train:
	# 		data_path, sub_folder, textgrid_folder, \
	# 		wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
	# 		phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)
    #
	# 		writeSegments(full_path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
	# 					  data_path=data_path,
	# 					  sub_folder=sub_folder.replace('/', '_'),
	# 					  rl=filename,
	# 					  tier0=line_tier,
	# 					  tier1=syllable_tier,
	# 					  roletype=roletype)
    #
	# with open(os.path.join(path_data_test, 'segments'), "w") as f:
	# 	for rec in recordings_test:
	# 		data_path, sub_folder, textgrid_folder, \
	# 		wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
	# 		phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)
    #
	# 		writeSegments(full_path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
	# 					  data_path=data_path,
	# 					  sub_folder=sub_folder.replace('/', '_'),
	# 					  rl=filename,
	# 					  tier0=line_tier,
	# 					  tier1=syllable_tier,
	# 					  roletype=roletype)
    #
	# write text
	# with open(os.path.join(path_lang, "dict_lexicon_repetition.json"), "r") as read_file:
	# 	dict_lexicon = json.load(read_file)

	# with open(os.path.join(path_data_train, 'text'), "w") as f:
	# 	for rec in recordings_train:
	# 		data_path, sub_folder, textgrid_folder, \
	# 		wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
	# 		phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)
    #
	# 		writeText(full_path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
	# 				  data_path=data_path,
	# 				  sub_folder=sub_folder.replace('/', '_'),
	# 				  rl=filename,
	# 				  line_tier=line_tier,
	# 				  syllable_tier=special_tier,
	# 				  phoneme_tier=phoneme_tier,
	# 				  roletype=roletype,
	# 				  dict_lexicon=dict_lexicon)

	# # test set alignment text
	# with open(os.path.join(path_data_test, 'text'), "w") as f:
	# 	for rec in recordings_test:
	# 		data_path, sub_folder, textgrid_folder, \
	# 		wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
	# 		phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)
    #
	# 		writeText(full_path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
	# 				  data_path=data_path,
	# 				  sub_folder=sub_folder.replace('/', '_'),
	# 				  rl=filename,
	# 				  line_tier=line_tier,
	# 				  syllable_tier=syllable_tier,
	# 				  phoneme_tier=phoneme_tier,
	# 				  roletype=roletype,
	# 				  dict_lexicon=None)

	# test set teacher ground truth text
	with open(os.path.join(path_data_test, 'text_teacher'), "w") as f:
		for rec in recordings_test:
			data_path, sub_folder, textgrid_folder, \
			wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
			phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)

			writeText(full_path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
					  data_path=data_path,
					  sub_folder=sub_folder.replace('/', '_'),
					  rl=filename,
					  line_tier=line_tier,
					  syllable_tier="specialTeacher",
					  phoneme_tier=phoneme_tier,
					  roletype=roletype,
					  dict_lexicon=None)

	# test set student ground truth text
	with open(os.path.join(path_data_test, 'text_student'), "w") as f:
		for rec in recordings_test:
			data_path, sub_folder, textgrid_folder, \
			wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
			phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)

			writeText(full_path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
					  data_path=data_path,
					  sub_folder=sub_folder.replace('/', '_'),
					  rl=filename,
					  line_tier=line_tier,
					  syllable_tier="special",
					  phoneme_tier=phoneme_tier,
					  roletype=roletype,
					  dict_lexicon=None)

	# # write wav.scp
	# with open(os.path.join(path_data_train, 'wav.scp'), "w") as f:
	# 	for rec in recordings_train:
	# 		data_path, sub_folder, textgrid_folder, \
	# 		wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
	# 		phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)
    #
	# 		writeWavScp(roletype=roletype,
	# 					data_path=data_path,
	# 					sub_folder=sub_folder.replace('/', '_'),
	# 					full_path_wav=os.path.join(path_root, data_path, wav_folder, sub_folder, filename+'.wav'),
	# 					rl=filename)
    #
	# with open(os.path.join(path_data_test, 'wav.scp'), "w") as f:
	# 	for rec in recordings_test:
	# 		data_path, sub_folder, textgrid_folder, \
	# 		wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
	# 		phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)
    #
	# 		writeWavScp(roletype=roletype,
	# 					data_path=data_path,
	# 					sub_folder=sub_folder.replace('/', '_'),
	# 					full_path_wav=os.path.join(path_root, data_path, wav_folder, sub_folder, filename+'.wav'),
	# 					rl=filename)
    #
	# # write utt2spk
	# with open(os.path.join(path_data_train, 'utt2spk'), "w") as f:
	# 	for rec in recordings_train:
	# 		data_path, sub_folder, textgrid_folder, \
	# 		wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
	# 		phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)
    #
	# 		writeUtt2spk(full_path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
	# 					 data_path=data_path,
	# 					 sub_folder=sub_folder.replace('/', '_'),
	# 					 rl=filename,
	# 					 line_tier=line_tier,
	# 					 syllable_tier=special_tier,
	# 					 roletype=roletype)
    #
	# with open(os.path.join(path_data_test, 'utt2spk'), "w") as f:
	# 	for rec in recordings_test:
	# 		data_path, sub_folder, textgrid_folder, \
	# 		wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
	# 		phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)
    #
	# 		writeUtt2spk(full_path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
	# 					 data_path=data_path,
	# 					 sub_folder=sub_folder.replace('/', '_'),
	# 					 rl=filename,
	# 					 line_tier=line_tier,
	# 					 syllable_tier=special_tier,
	# 					 roletype=roletype)
    #
	# # write spk2utt
	# utt_list_dan = ["Dan", " "]
	# utt_list_laosheng = ["Laosheng", " "]
    #
	# for rec in recordings_train:
	# 	data_path, sub_folder, textgrid_folder, \
	# 	wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
	# 	phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)
    #
	# 	if roletype == "Dan":
	# 		writeSpk2utt(full_path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
	# 					 data_path=data_path,
	# 					 sub_folder=sub_folder.replace('/', '_'),
	# 					 rl=filename,
	# 					 line_tier=line_tier,
	# 					 syllable_tier=special_tier,
	# 					 roletype=roletype,
	# 					 utt_list=utt_list_dan)
	# 	else:
	# 		writeSpk2utt(full_path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
	# 					 data_path=data_path,
	# 					 sub_folder=sub_folder.replace('/', '_'),
	# 					 rl=filename,
	# 					 line_tier=line_tier,
	# 					 syllable_tier=special_tier,
	# 					 roletype=roletype,
	# 					 utt_list=utt_list_laosheng)
    #
	# with open(os.path.join(path_data_train, 'spk2utt'), "w") as f:
	# 	f.write(''.join(utt_list_dan))
	# 	f.write('\n')
	# 	f.write(''.join(utt_list_laosheng))
	# 	f.write('\n')
    #
	# utt_list_dan = ["Dan", " "]
	# utt_list_laosheng = ["Laosheng", " "]
    #
	# for rec in recordings_test:
	# 	data_path, sub_folder, textgrid_folder, \
	# 	wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
	# 	phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)
    #
	# 	if roletype == "Dan":
	# 		writeSpk2utt(full_path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
	# 					 data_path=data_path,
	# 					 sub_folder=sub_folder.replace('/', '_'),
	# 					 rl=filename,
	# 					 line_tier=line_tier,
	# 					 syllable_tier=special_tier,
	# 					 roletype=roletype,
	# 					 utt_list=utt_list_dan)
	# 	else:
	# 		writeSpk2utt(full_path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
	# 					 data_path=data_path,
	# 					 sub_folder=sub_folder.replace('/', '_'),
	# 					 rl=filename,
	# 					 line_tier=line_tier,
	# 					 syllable_tier=special_tier,
	# 					 roletype=roletype,
	# 					 utt_list=utt_list_laosheng)
    #
	# with open(os.path.join(path_data_test, 'spk2utt'), "w") as f:
	# 	f.write(''.join(utt_list_dan))
	# 	f.write('\n')
	# 	f.write(''.join(utt_list_laosheng))
	# 	f.write('\n')
    #
	# # write phone.txt
	# with open(os.path.join(path_data_train, 'phone.txt'), "w") as f:
	# 	for rec in recordings_train:
	# 		data_path, sub_folder, textgrid_folder, \
	# 		wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
	# 		phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)
    #
	# 		writePhone(full_path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
	# 				   data_path=data_path,
	# 				   sub_folder=sub_folder,
	# 				   rl=filename,
	# 				   line_tier=line_tier,
	# 				   phoneme_tier=phoneme_tier,
	# 				   roletype=roletype)
    #
	# with open(os.path.join(path_data_test, 'phone.txt'), "w") as f:
	# 	for rec in recordings_test:
	# 		data_path, sub_folder, textgrid_folder, \
	# 		wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
	# 		phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)
    #
	# 		writePhone(full_path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
	# 				   data_path=data_path,
	# 				   sub_folder=sub_folder,
	# 				   rl=filename,
	# 				   line_tier=line_tier,
	# 				   phoneme_tier=phoneme_tier,
	# 				   roletype=roletype)

