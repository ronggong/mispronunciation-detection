import os


def parse_text_to_dict(filename):
    dict_text = {}
    with open(filename) as file:
        for row in file.readlines():
            row = row.replace('\n', '')
            key = row.split(' ')[0]
            val = row.split(' ')[1:]
            dict_text[key] = val
    return dict_text


if __name__ == '__main__':
    path_test_ali = "/Users/ronggong/PycharmProjects/mispronunciation-detection/kaldi_alignment/exp/mono_test_ali/"
    path_test = "/Users/ronggong/PycharmProjects/mispronunciation-detection/kaldi_alignment/data/test"

    filename_groundtruth_teacher = os.path.join(path_test, 'text_teacher')
    filename_groundtruth_student = os.path.join(path_test, 'text_student')
    filename_decoded_student = os.path.join(path_test_ali, 'text_decoded')

    dict_groundtruth_teacher = parse_text_to_dict(filename_groundtruth_teacher)
    dict_groundtruth_student = parse_text_to_dict(filename_groundtruth_student)
    dict_decoded_student = parse_text_to_dict(filename_decoded_student)

    special_correct_stu = 0
    special_mis_stu = 0
    jianzi_correct_stu = 0
    jianzi_mis_stu = 0

    special_correct_tea, special_mis_tea, jianzi_correct_tea, jianzi_mis_tea = 0, 0, 0, 0
    for key in dict_decoded_student:
        val_decoded_stu = dict_decoded_student[key]
        val_gt_tea = dict_groundtruth_teacher[key]
        val_gt_tea_syl = [val for ii, val in enumerate(val_gt_tea) if ii % 2.0 == 0]
        val_gt_tea_class = [val for ii, val in enumerate(val_gt_tea) if ii % 2.0 != 0]
        val_gt_stu = dict_groundtruth_student[key]

        assert len(val_gt_tea_syl) == len(val_gt_stu) == len(val_decoded_stu)

        for ii, syl_class in enumerate(val_gt_tea_class):
            if syl_class == '1':
                if val_gt_stu[ii] == val_decoded_stu[ii]:
                    special_correct_stu += 1
                else:
                    special_mis_stu += 1
            elif syl_class == '2':
                if val_gt_stu[ii] == val_decoded_stu[ii]:
                    jianzi_correct_stu += 1
                else:
                    jianzi_mis_stu += 1

        for ii, syl_class in enumerate(val_gt_tea_class):
            if syl_class == '1':
                if val_gt_tea_syl[ii] == val_gt_stu[ii]:
                    if val_gt_tea_syl[ii] == val_decoded_stu[ii]:
                        special_correct_tea += 1
                    else:
                        special_mis_tea += 1
                else:
                    if val_gt_tea_syl[ii] == val_decoded_stu[ii]:
                        special_mis_tea += 1
                    else:
                        special_correct_tea += 1
            elif syl_class == '2':
                if val_gt_tea_syl[ii] == val_gt_stu[ii]:
                    if val_gt_tea_syl[ii] == val_decoded_stu[ii]:
                        jianzi_correct_tea += 1
                    else:
                        jianzi_mis_tea += 1
                else:
                    if val_gt_tea_syl[ii] == val_decoded_stu[ii]:
                        jianzi_mis_tea += 1
                    else:
                        jianzi_correct_tea += 1

    print(special_correct_stu, special_mis_stu, jianzi_correct_stu, jianzi_mis_stu)
    print(special_correct_tea, special_mis_tea, jianzi_correct_tea, jianzi_mis_tea)

