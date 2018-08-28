import os
from neural_net.utils.textgridParser import textGrid2WordList
from neural_net.utils.textgridParser import wordListsParseByLines


def parse_syllable_line_list(ground_truth_text_grid_file, parent_tier, child_tier):

    if not os.path.isfile(ground_truth_text_grid_file):
        is_file_exist = False
        return False, is_file_exist, False
    else:
        is_file_exist = True

        # parse line
        line_list, _ = textGrid2WordList(ground_truth_text_grid_file, whichTier=parent_tier)

        # parse syllable
        syllable_list, is_syllable_found = textGrid2WordList(ground_truth_text_grid_file, whichTier=child_tier)

        # parse lines of ground truth
        nested_syllable_lists, _, _ = wordListsParseByLines(line_list, syllable_list)

        return nested_syllable_lists, is_file_exist, is_syllable_found
