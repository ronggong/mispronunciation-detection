import pickle
import numpy as np
import soundfile as sf
from neural_net.file_path import *
from neural_net.parameters import *
from neural_net.utils.audio_preprocessing import get_log_mel_madmom
from neural_net.utils.audio_preprocessing import feature_reshape
from neural_net.utils.utils_functions import smooth_obs
from neural_net.onsetSegmentEval.evaluation import onsetEval
from neural_net.onsetSegmentEval.evaluation import segment_eval_helper
from neural_net.onsetSegmentEval.evaluation import segmentEval
from neural_net.onsetSegmentEval.evaluation import metrics
from neural_net.training_scripts.attention import Attention
from neural_net.training_scripts.models_RNN import RNN_model_definition
from neural_net.plot_code import plot_spectro_att
from kaldi_alignment.srcPy.textgridParser import syllableTextgridExtraction
from keras.models import load_model

import pyximport
pyximport.install(reload_support=True,
                  setup_args={'include_dirs': np.get_include()})

import viterbiDecodingPhonemeSeg


def duration_constraint(dur, ratio):
    if dur <= 0:
        dur = 0.01
    elif dur > dur * (1.0 + 3 * ratio * dur):
        dur = dur * (1.0 + 3 * ratio * dur)
    else:
        pass
    return dur


def parse_text_to_dict(filename):
    dict_text = {}
    with open(filename) as file:
        for row in file.readlines():
            row = row.replace('\n', '')
            key = row.split(' ')[0]
            val = row.split(' ')[1:]
            dict_text[key] = val
    return dict_text


def special_jianzi_detection(full_path_textgrid,
                             full_path_wav,
                             roletype,
                             data_path,
                             sub_folder,
                             rn,
                             line_tier,
                             model_special,
                             model_jianzi,
                             model_joint,
                             scaler_joint,
                             f,
                             ratio=0.1,
                             plot_jianzi_att=False):

    wav_file = os.path.join(full_path_wav, rn+".wav")

    # get wav duration
    data_wav, fs_wav = sf.read(wav_file)

    # calculate log mel feature
    log_mel = get_log_mel_madmom(wav_file, fs=fs_wav, hopsize_t=hopsize_t, channel=1, context=True)
    log_mel = scaler_joint.transform(log_mel)
    log_mel = feature_reshape(log_mel, nlen=7)

    log_mel_no_context = get_log_mel_madmom(wav_file, fs=fs_wav, hopsize_t=hopsize_t, channel=1, context=False)

    nestedSyllableClassLists, numLines, numSyllables = \
        syllableTextgridExtraction(full_path_textgrid, rn, line_tier, "specialClassTeacher")

    sum_num_detected_onset, sum_num_gt_onset, sum_num_correct_onset = 0, 0, 0
    sum_sample_correct, sum_sample_total = 0, 0

    ii = 0
    for ii_line, line in enumerate(nestedSyllableClassLists):

        if not len(line[1]):
            continue

        start_frame = int(round(line[0][0] / hopsize_t))
        end_frame = int(round(line[0][1] / hopsize_t))

        log_mel_line = log_mel[start_frame: end_frame]
        log_mel_line = np.expand_dims(log_mel_line, axis=1)

        log_mel_line_no_context = log_mel_no_context[start_frame: end_frame]

        obs_syllable, _ = model_joint.predict(log_mel_line, batch_size=128, verbose=2)

        obs_syllable = np.squeeze(obs_syllable)

        obs_syllable = smooth_obs(obs_syllable)

        obs_syllable[0] = 1.0
        obs_syllable[-1] = 1.0

        # prepare the duration
        list_duration = np.array([syl[1] - syl[0] for syl in line[1] if len(syl[2])])
        # random sample the duration from a normal distribution
        list_duration = np.random.normal(loc=list_duration, scale=ratio*list_duration)
        # constraint the durations
        for ii_dur in range(len(list_duration)):
            list_duration[ii_dur] = duration_constraint(dur=list_duration[ii_dur], ratio=ratio)

        list_duration *= (line[0][1] - line[0][0]) / np.sum(list_duration)

        list_syl = [syl[2] for syl in line[1] if len(syl[2])]

        boundaries_syllable = viterbiDecodingPhonemeSeg.viterbiSegmental2(obs_syllable, list_duration, varin)
        boundaries_syllable_start_time = np.array(boundaries_syllable[:-1]) * hopsize_t

        list_gt_onset = [[syl[0]-line[1][0][0], syl[2]] for syl in line[1] if len(syl[2])]
        list_decoded_onset = [[boundaries_syllable_start_time[ii_syl], list_syl[ii_syl]]
                              for ii_syl in range(len(list_syl))]

        # onset detection eval
        numDetectedOnsets, numGroundtruthOnsets, \
        numOnsetCorrect, _, _ = onsetEval(groundtruthOnsets=list_gt_onset,
                                          detectedOnsets=list_decoded_onset,
                                          tolerance=0.025,
                                          label=False)
        sum_num_detected_onset += numDetectedOnsets
        sum_num_gt_onset += numGroundtruthOnsets
        sum_num_correct_onset += numOnsetCorrect

        # segmentation eval
        syllable_gt_onset_resample = segment_eval_helper(list_gt_onset, line_time=line[0][1] - line[0][0])
        syllable_decoded_onset_resample = segment_eval_helper(list_decoded_onset, line_time=line[0][1] - line[0][0])
        sample_correct, sample_total = \
            segmentEval(syllable_gt_onset_resample, syllable_decoded_onset_resample)
        sum_sample_correct += sample_correct
        sum_sample_total += sample_total

        # add 0s to sentence number
        zero_adding = ''
        for jj in range(3 - len(str(ii))):
            zero_adding += '0'
        f.write(roletype + '_' + data_path + '_' + sub_folder + '_' + rn + '_' + zero_adding + str(ii) + ' ')
        ii_syl_boundary = 0
        for ii_syl, syl in enumerate(line[1]):
            if len(syl[2]):
                special_class_teacher = nestedSyllableClassLists[ii_line][1][ii_syl][2]
                if special_class_teacher == "1" or special_class_teacher == "2":
                    log_mel_syl = \
                        log_mel_line_no_context[boundaries_syllable[ii_syl_boundary]: boundaries_syllable[ii_syl_boundary+1]]
                    log_mel_syl = np.expand_dims(log_mel_syl, axis=0)
                if special_class_teacher == '1':
                    y_pred = model_special.predict_on_batch(log_mel_syl)
                    pred = "1" if y_pred[0][0] > 0.5 else "0"
                    f.write(pred)
                    print("debug {}".format(y_pred[0][0]))
                elif special_class_teacher == '2':
                    y_pred = model_jianzi.predict_on_batch(log_mel_syl)
                    pred = "1" if y_pred[0][0][0] > 0.5 else "0"
                    f.write(pred)
                    print("debug {}, len att vector {}, len log mel {}, syl {}".
                          format(y_pred[0][0][0], len(y_pred[1][0]), log_mel_syl.shape[1], syl[2].strip()))
                    if plot_jianzi_att:
                        filename_save = \
                            os.path.join(path_figs_jianzi,
                                         syl[2].strip().upper()+'_'+str(ii)+'_'+str(ii_syl_boundary)+'.png')
                        plot_spectro_att(mfcc0=log_mel_syl[0,:,:],
                                         att_vector=y_pred[1][0],
                                         hopsize_t=hopsize_t,
                                         filename_save=filename_save)
                else:
                    f.write(syl[2].strip().upper())
                ii_syl_boundary += 1
            else:
                pass
            if len(syl[2]):
                if ii_syl != len(line[1]) - 1:
                    f.write(' ')

        f.write('\n')
        ii += 1

    print("debug {}, {}, {}, {}, {}"
          .format(sum_num_correct_onset,
                  sum_num_detected_onset,
                  sum_num_gt_onset,
                  sum_sample_correct,
                  sum_sample_total))

    return sum_num_correct_onset, sum_num_detected_onset, sum_num_gt_onset, sum_sample_correct, sum_sample_total


def convert_decoded_text(full_path_textgrid, rn, data_path, sub_folder, line_tier, decoded_lines, jj_entire_line, corrected_lines):

    nestedSyllableClassLists, numLines, numSyllables = \
        syllableTextgridExtraction(full_path_textgrid, rn, line_tier, "specialClassTeacher")

    ii = 0
    for ii_line, line in enumerate(nestedSyllableClassLists):

        if not len(line[1]):
            continue

        # add 0s to sentence number
        zero_adding = ''
        for jj in range(3 - len(str(ii))):
            zero_adding += '0'

        line_head = [roletype + '_' + data_path + '_' + sub_folder + '_' + rn + '_' + zero_adding + str(ii)]
        line_tail = decoded_lines[jj_entire_line].split()[1:]

        corrected_lines.append(line_head + line_tail)

        ii += 1
        jj_entire_line += 1

    return jj_entire_line


def evaluation():
    path_test = "/Users/ronggong/PycharmProjects/mispronunciation-detection/kaldi_alignment/data/test"
    path_results = "/Users/ronggong/PycharmProjects/mispronunciation-detection/neural_net/results"
    filename_groundtruth_teacher = os.path.join(path_test, 'text_teacher')
    filename_groundtruth_student = os.path.join(path_test, 'text_student')
    filename_decoded_student = filename_result_decoded_mispronunciaiton
    dict_groundtruth_teacher = parse_text_to_dict(filename_groundtruth_teacher)
    dict_groundtruth_student = parse_text_to_dict(filename_groundtruth_student)
    dict_decoded_student = parse_text_to_dict(filename_decoded_student)

    special_correct_tea, special_mis_tea, jianzi_correct_tea, jianzi_mis_tea = 0, 0, 0, 0
    for key in dict_decoded_student:
        val_decoded_stu = dict_decoded_student[key]
        val_gt_tea = dict_groundtruth_teacher[key]
        val_gt_tea_syl = [val for ii, val in enumerate(val_gt_tea) if ii % 2.0 == 0]
        val_gt_tea_class = [val for ii, val in enumerate(val_gt_tea) if ii % 2.0 != 0]
        val_gt_stu = dict_groundtruth_student[key]

        assert len(val_gt_tea_syl) == len(val_decoded_stu)

        for ii, syl_class in enumerate(val_gt_tea_class):
            if syl_class == '1':
                if val_gt_tea_syl[ii] == val_gt_stu[ii]:
                    if val_decoded_stu[ii] == "1":
                        special_correct_tea += 1
                    else:
                        special_mis_tea += 1
                else:
                    if val_decoded_stu[ii] == "1":
                        special_mis_tea += 1
                    else:
                        special_correct_tea += 1
            elif syl_class == '2':
                if val_gt_tea_syl[ii] == val_gt_stu[ii]:
                    if val_decoded_stu[ii] == "1":
                        jianzi_correct_tea += 1
                    else:
                        jianzi_mis_tea += 1
                else:
                    if val_decoded_stu[ii] == "1":
                        jianzi_mis_tea += 1
                    else:
                        jianzi_correct_tea += 1

    print(special_correct_tea, special_mis_tea, jianzi_correct_tea, jianzi_mis_tea)


def decoded_text_correction():
    path_results = "/Users/ronggong/PycharmProjects/mispronunciation-detection/neural_net/results"
    filename_decoded_student = os.path.join(path_results, 'text_decoded_special_True_True_0.5')

    with open(filename_decoded_student, "r") as f:
        lines = f.readlines()

    corrected_lines = []
    jj_entire_line = 0
    for rec in recordings_test:
        data_path, sub_folder, textgrid_folder, \
        wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
        phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)

        jj_entire_line = convert_decoded_text(full_path_textgrid=os.path.join(path_root,
                                                                              data_path,
                                                                              textgrid_folder,
                                                                              sub_folder),
                                              rn=filename,
                                              data_path=data_path,
                                              sub_folder=sub_folder.replace("/", "_"),
                                              line_tier=line_tier,
                                              decoded_lines=lines,
                                              jj_entire_line=jj_entire_line,
                                              corrected_lines=corrected_lines)

    with open(filename_decoded_student+"_corrected", "w") as f:
        for line in corrected_lines:
            f.write(" ".join(line)+"\n")


if __name__ == "__main__":

    decode_eval = "decode"
    plot_jianzi_att = True

    if decode_eval == "decode":
        sum_num_correct_onset, sum_num_detected_onset, sum_num_gt_onset = 0, 0, 0
        sum_sample_correct, sum_sample_total = 0, 0

        model_special = load_model(filepath=filename_special_model,
                                   custom_objects={'Attention': Attention(return_attention=True)})
        model_jianzi = load_model(filepath=filename_jianzi_model,
                                  custom_objects={'Attention': Attention(return_attention=True)})

        # load weights from the pre-trained model
        weights_jianzi_model = model_jianzi.get_weights()

        # redefine the model to extract the attention vector
        batch_size = 1
        input_shape = (batch_size, None, 80)
        patience = 15
        attention = "feedforward"
        conv = True
        dropout = 0.5

        model_jianzi_redefined, _, att_vector = \
            RNN_model_definition(input_shape=input_shape,
                                 conv=conv,
                                 dropout=dropout,
                                 attention=attention,
                                 output_shape=1)

        model_jianzi_redefined.set_weights(weights_jianzi_model)

        model_jianzi_redefined.summary()

        # model_special = load_model(filepath=filename_special_model)
        # model_jianzi = load_model(filepath=filename_jianzi_model)

        # load keras joint cnn model
        model_joint = load_model(os.path.join(joint_cnn_model_path, 'jan_joint0.h5'))

        # load log mel feature scaler
        scaler_joint = pickle.load(open(os.path.join(joint_cnn_model_path, 'scaler_joint.pkl'), 'rb'),
                                   encoding='latin1')

        with open(filename_result_decoded_mispronunciaiton, "w") as f:
            for rec in recordings_test:
                data_path, sub_folder, textgrid_folder, \
                wav_folder, filename, line_tier, longsyllable_tier, syllable_tier, \
                phoneme_tier, special_tier, special_class_tier, roletype = parse_recordings(rec)

                num_correct_onset, num_detected_onset, num_gt_onset, sample_correct, sample_total = \
                    special_jianzi_detection(full_path_textgrid=os.path.join(path_root, data_path, textgrid_folder, sub_folder),
                                             full_path_wav=os.path.join(path_root, data_path, wav_folder, sub_folder),
                                             roletype=roletype,
                                             data_path=data_path,
                                             sub_folder=sub_folder.replace("/", "_"),
                                             rn=filename,
                                             line_tier=line_tier,
                                             model_special=model_special,
                                             model_jianzi=model_jianzi_redefined,
                                             model_joint=model_joint,
                                             scaler_joint=scaler_joint,
                                             f=f,
                                             plot_jianzi_att=plot_jianzi_att)
                sum_num_correct_onset += num_correct_onset
                sum_num_detected_onset += num_detected_onset
                sum_num_gt_onset += num_gt_onset
                sum_sample_correct += sample_correct
                sum_sample_total += sample_total

        recall_onset, precision_onset, F1_onset = metrics(numCorrect=sum_num_correct_onset,
                                                          numDetected=sum_num_detected_onset,
                                                          numGroundtruth=sum_num_gt_onset)

        acc_syllable = sum_sample_correct / float(sum_sample_total)

        print("Recall, precision, F1: {} {} {}".format(recall_onset, precision_onset, F1_onset))
        print("Segmentation accuracy, {}".format(acc_syllable*100.0))
    elif decode_eval == "eval":
        evaluation()


