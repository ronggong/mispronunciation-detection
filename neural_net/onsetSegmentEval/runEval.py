"""run evaluation on the onset detection results, output results in the eval directory"""


# from parameters import *
import os
import pickle
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from filePath import getRecordingNames
from filePath import getRecordings
from filePath import path_root
from filePath import path_primary
from evaluation import onsetEval
from evaluation import segmentEval
from evaluation import metrics
from utils import textgrid_syllable_phoneme_parser
from parseAli import convert_utt_list_2_student_list_test

hopsize_t = 0.01

def write_results_2_txt(filename,
                        results):
    """write the evaluation results to a text file"""

    with open(filename, 'w') as f:

        # no label 0.025
        f.write(str(results[0]))
        f.write('\n')
        f.write(str(results[1]))
        f.write('\n')
        f.write(str(results[2]))
        f.write('\n')

        # no label 0.05
        f.write(str(results[3]))
        f.write('\n')
        f.write(str(results[4]))
        f.write('\n')
        f.write(str(results[5]))
        f.write('\n')

        # label 0.025
        f.write(str(results[6]))
        f.write('\n')
        f.write(str(results[7]))
        f.write('\n')
        f.write(str(results[8]))
        f.write('\n')

        # label 0.05
        f.write(str(results[9]))
        f.write('\n')
        f.write(str(results[10]))
        f.write('\n')
        f.write(str(results[11]))


def batch_run_metrics_calculation(sumStat, gt_onsets, detected_onsets):
    """
    Batch run the metric calculation
    :param sumStat:
    :param gt_onsets:
    :param detected_onsets:
    :return:
    """
    counter = 0
    for l in [False, True]:
        for t in [0.025, 0.05]:
            numDetectedOnsets, numGroundtruthOnsets, \
            numOnsetCorrect, _, _ = onsetEval(gt_onsets, detected_onsets, t, l)

            sumStat[counter, 0] += numDetectedOnsets
            sumStat[counter, 1] += numGroundtruthOnsets
            sumStat[counter, 2] += numOnsetCorrect

            counter += 1


def metrics_aggregation(sumStat):

    recall_nolabel_25, precision_nolabel_25, F1_nolabel_25 = metrics(sumStat[0, 0], sumStat[0, 1], sumStat[0, 2])
    recall_nolabel_5, precision_nolabel_5, F1_nolabel_5 = metrics(sumStat[1, 0], sumStat[1, 1], sumStat[1, 2])
    recall_label_25, precision_label_25, F1_label_25 = metrics(sumStat[2, 0], sumStat[2, 1], sumStat[2, 2])
    recall_label_5, precision_label_5, F1_label_5 = metrics(sumStat[3, 0], sumStat[3, 1], sumStat[3, 2])

    return precision_nolabel_25, recall_nolabel_25, F1_nolabel_25, \
           precision_nolabel_5, recall_nolabel_5, F1_nolabel_5, \
           precision_label_25, recall_label_25, F1_label_25, \
           precision_label_5, recall_label_5, F1_label_5


def run_eval_onset(role_type, artist_name, aria_name, student_name, utt_num, start_time_label_syl, start_time_label_phn, method, ali_rec='ali', mono_tri_str='mono'):
    """
    run evaluation for onset detection
    :param method: hsmm or joint:
    :param param_str different configurations and save the results to different txt files
    :param test_val: string, val or test evaluate for validation or test file
    :return:
    """

    sumStat_syllable = np.zeros((4, 3), dtype='int')
    sumStat_phoneme = np.zeros((4, 3), dtype='int')

    for ii in range(len(student_name)):
        fn_textgrid_gt = os.path.join(path_root, path_primary, 'textgrid', artist_name[ii], aria_name[ii], student_name[ii]+'.textgrid')

        gt_syllable_lists, gt_phoneme_lists = \
            textgrid_syllable_phoneme_parser(fn_textgrid_gt, 'dianSilence', 'details')

        for jj in range(len(gt_syllable_lists)):
            syllable_gt_onsets = [[unit[0]-gt_syllable_lists[jj][0][0], unit[2]] for unit in gt_syllable_lists[jj][1] if len(unit[2])]
            phoneme_gt_onsets = [[unit[0]-gt_phoneme_lists[jj][0][0], unit[2]] for unit in gt_phoneme_lists[jj][1] if len(unit[2])]

            try:
                syllable_detected_onsets = start_time_label_syl[ii][jj]
                phoneme_detected_onsets = start_time_label_phn[ii][jj]

                # if len(syllable_gt_onsets) != len(syllable_detected_onsets):
                #     print(fn_textgrid_gt)
                #     print(syllable_gt_onsets)
                #     print(syllable_detected_onsets)
                #     print(phoneme_gt_onsets)
                #     print(phoneme_detected_onsets)

                batch_run_metrics_calculation(sumStat_syllable, syllable_gt_onsets, syllable_detected_onsets)
                batch_run_metrics_calculation(sumStat_phoneme, phoneme_gt_onsets, phoneme_detected_onsets)
            except IndexError:
                print(ii, jj)

    result_syllable = metrics_aggregation(sumStat_syllable)
    result_phoneme = metrics_aggregation(sumStat_phoneme)

    current_path = os.path.dirname(os.path.abspath(__file__))

    write_results_2_txt(os.path.join(current_path, '../..', 'results_eval', 'syllable_'+ali_rec+'_'+mono_tri_str+'_onset_'+method+'.txt'),
                        result_syllable)

    write_results_2_txt(os.path.join(current_path, '../..', 'results_eval', 'phoneme_'+ali_rec+'_'+mono_tri_str+'_onset_'+method+'.txt'),
                        result_phoneme)

    return result_phoneme[2], result_phoneme[8]


def segment_eval_helper(onsets, line_time):
    onsets_frame = np.round(np.array([sgo[0] for sgo in onsets]) / hopsize_t)

    resample = [onsets[0][1]]

    current = onsets[0][1]

    for ii_sample in range(1, int(round(line_time / hopsize_t))):

        if ii_sample in onsets_frame:
            idx_onset = np.where(onsets_frame == ii_sample)
            idx_onset = idx_onset[0][0]
            current = onsets[idx_onset][1]
        resample.append(current)

    return resample


def run_eval_segment(role_type, artist_name, aria_name, student_name, utt_num, start_time_label_syl, start_time_label_phn, method, ali_rec='ali', mono_tri_str='mono'):
    """segment level evaluation"""
    

    sumSampleCorrect_syllable, sumSampleCorrect_phoneme, \
    sumSample_syllable, sumSample_phoneme = 0,0,0,0

    for ii in range(len(student_name)):
        fn_textgrid_gt = os.path.join(path_root, path_primary, 'textgrid', artist_name[ii], aria_name[ii], student_name[ii]+'.textgrid')

        gt_syllable_lists, gt_phoneme_lists = \
            textgrid_syllable_phoneme_parser(fn_textgrid_gt, 'dianSilence', 'details')

        for jj in range(len(gt_syllable_lists)):
            line_time = gt_syllable_lists[jj][0][1] - gt_syllable_lists[jj][0][0]
            syllable_gt_onsets = [[unit[0]-gt_syllable_lists[jj][0][0], unit[2]] for unit in gt_syllable_lists[jj][1] if len(unit[2])]
            phoneme_gt_onsets = [[unit[0]-gt_phoneme_lists[jj][0][0], unit[2]] for unit in gt_phoneme_lists[jj][1] if len(unit[2])]

            try:
                syllable_detected_onsets = start_time_label_syl[ii][jj]
                phoneme_detected_onsets = start_time_label_phn[ii][jj]

                syllable_gt_onsets_resample = segment_eval_helper(syllable_gt_onsets, line_time)
                syllable_detected_onsets_resample = segment_eval_helper(syllable_detected_onsets, line_time)
                phoneme_gt_onsets_resample = segment_eval_helper(phoneme_gt_onsets, line_time)
                phoneme_detected_onsets_resample = segment_eval_helper(phoneme_detected_onsets, line_time)

                sample_correct_syllable, sample_syllable = \
                    segmentEval(syllable_gt_onsets_resample, syllable_detected_onsets_resample)
                sample_correct_phoneme, sample_phoneme = \
                    segmentEval(phoneme_gt_onsets_resample, phoneme_detected_onsets_resample)

                sumSampleCorrect_syllable += sample_correct_syllable
                sumSampleCorrect_phoneme += sample_correct_phoneme
                sumSample_syllable += sample_syllable
                sumSample_phoneme += sample_phoneme
            except IndexError:
                print(ii, jj)

    acc_syllable = sumSampleCorrect_syllable/float(sumSample_syllable)
    acc_phoneme = sumSampleCorrect_phoneme/float(sumSample_phoneme)

    acc_syllable *= 100
    acc_phoneme *= 100

    # write the results to txt only in test mode
    current_path = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(current_path, '../..', 'results_eval', 'syllable_'+ali_rec+'_'+mono_tri_str+'_segment_'+method+'.txt'), 'w') as f:
        f.write(str(acc_syllable))

    with open(os.path.join(current_path, '../..', 'results_eval', 'phoneme_'+ali_rec+'_'+mono_tri_str+'_segment_'+method+'.txt'), 'w') as f:
        f.write(str(acc_phoneme))

    return acc_syllable, acc_phoneme


def eval_alignment(path_kaldi, mono):
    if mono:
        mono_tri_str = 'mono'
    else:
        mono_tri_str = 'tri3a'

    fn_ali_syl_dan = os.path.join(path_kaldi, 'exp/'+mono_tri_str+'_test_ali/ctm.1')
    fn_ali_syl_laosheng = os.path.join(path_kaldi, 'exp/'+mono_tri_str+'_test_ali/ctm.2')
    fn_ali_phn_dan = os.path.join(path_kaldi, 'exp/'+mono_tri_str+'_test_ali/ali.1.ctm')
    fn_ali_phn_laosheng = os.path.join(path_kaldi, 'exp/'+mono_tri_str+'_test_ali/ali.2.ctm')

    fn_phone = os.path.join(path_kaldi, 'exp/'+mono_tri_str+'_test_ali/phones.txt')

    role_type_syl_dan, artist_name_syl_dan, aria_name_syl_dan, student_name_syl_dan, utt_num_syl_dan, start_time_label_syl_dan = \
        convert_utt_list_2_student_list_test(fn_ali_syl_dan, fn_phone, 'syl')
    role_type_syl_ls, artist_name_syl_ls, aria_name_syl_ls, student_name_syl_ls, utt_num_syl_ls, start_time_label_syl_ls = \
        convert_utt_list_2_student_list_test(fn_ali_syl_laosheng, fn_phone, 'syl')

    role_type_syl = role_type_syl_dan + role_type_syl_ls
    artist_name_syl = artist_name_syl_dan + artist_name_syl_ls
    aria_name_syl = aria_name_syl_dan + aria_name_syl_ls
    student_name_syl = student_name_syl_dan + student_name_syl_ls
    utt_num_syl = utt_num_syl_dan + utt_num_syl_ls
    start_time_label_syl = start_time_label_syl_dan + start_time_label_syl_ls

    role_type_phn_dan, artist_name_phn_dan, aria_name_phn_dan, student_name_phn_dan, utt_num_phn_dan, start_time_label_phn_dan = \
        convert_utt_list_2_student_list_test(fn_ali_phn_dan, fn_phone, 'phn')
    role_type_phn_ls, artist_name_phn_ls, aria_name_phn_ls, student_name_phn_ls, utt_num_phn_ls, start_time_label_phn_ls = \
        convert_utt_list_2_student_list_test(fn_ali_phn_laosheng, fn_phone, 'phn')

    role_type_phn = role_type_phn_dan + role_type_phn_ls
    artist_name_phn = artist_name_phn_dan + artist_name_phn_ls
    aria_name_phn = aria_name_phn_dan + aria_name_phn_ls
    student_name_phn = student_name_phn_dan + student_name_phn_ls
    utt_num_phn = utt_num_phn_dan + utt_num_phn_ls
    start_time_label_phn = start_time_label_phn_dan + start_time_label_phn_ls

    run_eval_onset(role_type_syl, artist_name_syl, aria_name_syl, student_name_syl, utt_num_syl, start_time_label_syl, start_time_label_phn, 'flat_start', 'ali', mono_tri_str)

    run_eval_segment(role_type_syl, artist_name_syl, aria_name_syl, student_name_syl, utt_num_syl, start_time_label_syl, start_time_label_phn, 'flat_start', 'ali', mono_tri_str)


def eval_recognition(path_kaldi, LMWT, mono):
    if mono:
        mono_tri_str = 'mono'
    else:
        mono_tri_str = 'tri3a'
    fn_rec_syl = os.path.join(path_kaldi, 'exp/'+mono_tri_str+'/decode_test_word', 'score_'+str(LMWT), 'test.ctm')
    fn_rec_phn = os.path.join(path_kaldi, 'exp/'+mono_tri_str+'/decode_test_phone', 'score_'+str(LMWT), 'test_phone.ctm')
    fn_phone = os.path.join(path_kaldi, 'exp/'+mono_tri_str+'/phones.txt')

    role_type_syl, artist_name_syl, aria_name_syl, student_name_syl, utt_num_syl, start_time_label_syl = \
        convert_utt_list_2_student_list_test(fn_rec_syl, fn_phone, 'syl')

    role_type_phn, artist_name_phn, aria_name_phn, student_name_phn, utt_num_phn, start_time_label_phn = \
        convert_utt_list_2_student_list_test(fn_rec_phn, fn_phone, 'phn_label')

    run_eval_onset(role_type_syl, artist_name_syl, aria_name_syl, student_name_syl, utt_num_syl, start_time_label_syl, start_time_label_phn, 'score_'+str(LMWT), 'rec', mono_tri_str)

    run_eval_segment(role_type_syl, artist_name_syl, aria_name_syl, student_name_syl, utt_num_syl, start_time_label_syl, start_time_label_phn, 'score_'+str(LMWT), 'rec', mono_tri_str)



if __name__ == '__main__':

    path_kaldi = '/Users/ronggong/Documents_using/github/kaldi/egs/jingjuSinging'

    eval_alignment(path_kaldi, mono=False)
    # for ii in range(5, 21):
    #     eval_recognition(path_kaldi, ii)

    