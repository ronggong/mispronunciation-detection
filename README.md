# Mispronunciation detection
Mispronunciation detection code for jingju singing voice.

Rong Gong's thesis "Automatic assessment of singing voice pronunciation: 
a case study with jingju music" chapter 6.

This repo contains two methods:

* Baseline - forced alignment system built on Kaldi.
* Deep learning - discriminative model built using Keras and Tensorflow.

## Baseline
The main idea of the forced alignment-based mispronunciation detect is to use
two lexicons respectively for training and testing phases. The detail of this
idea is described in section 6.2.1 in the thesis.

We here only explain the general pipeline of the model training and testing. Please
write to the author if you want to know how to use the code for your own
experiment. Pipeline:

1. generate language dictionary by using `srcPy/parseLang.py`.
2. generate all the files that Kaldi need, e.g., text, wav.scp, phone.txt, by
`srcPy/parseTextRepetition.py`.
3. run the model training and decode the text for test data by `run.sh`
4. parse decoded pronunciation by `srcPy/parse_decoded_pronunciation.py`
5. evaluation `srcPy/mispron_eval.py`

## Deep learning-based discriminative model

We built discriminative models for mispronunciation detection. Two types of model
are built, one for special pronunciation, another for jiantuanzi syllables. We have
experimented several deep learning architectures, such as BiLSTM, CNN, attention, 
Temporal convolutional networks (TCNs), self-attention. The details are described in 
sections 6.3 and 6.4 in the thesis. Here, we also only describe the pipeline of model 
training and testing. Please write to the auther if you want to use the code for your own
experiment. Pipeline:

1. collecting training logarithmic Mel representation by `training_sample_collection_syllable.py`
2. train various deep learning architectures by using `train_rnn_jianzi.py` or `train_rnn_special.py` respectively for
special pronunciation and jiantuanzi models. e.g., attention var can be `feedforward` or `selfatt`.
3. train TCNs architectures by using `train_rnn_special_tcn.py` and `train_rnn_jianzi_tcn.py`
4. evaluation by `eval.py`

## Contact
Rong Gong: rong.gong<at>upf.edu

## Code license
GNU Affero General Public License 3.0
