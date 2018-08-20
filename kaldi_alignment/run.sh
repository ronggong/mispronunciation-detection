#!/bin/bash

exec 5> debug_output.txt
BASH_XTRACEFD="5"

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh

H=`pwd`  #exp home
n=2      #parallel jobs


for x in train test; do
   perl local/prepare_stm.pl data/$x || exit 1;
 	utils/fix_data_dir.sh data/$x || exit 1;
done

#produce MFCC features
rm -rf data/mfcc && mkdir -p data/mfcc &&  cp -R data/{train,test} data/mfcc || exit 1;
for x in train test; do
    #make  mfcc
    steps/make_mfcc.sh --nj $n --cmd "$train_cmd" data/mfcc/$x exp/make_mfcc/$x mfcc/$x || exit 1;
    #compute cmvn
    steps/compute_cmvn_stats.sh data/mfcc/$x exp/mfcc_cmvn/$x mfcc/$x || exit 1;
done

rm -rf data/local/ data/lang && rm -f data/dict/lexiconp.txt || exit 1;

# remove the language files for test data
rm -rf data/local_test/ data/lang_test && rm -f data/dict_test/lexiconp.txt || exit 1;

#lang
# prepare language stuffs for train data
utils/prepare_lang.sh --sil-prob 0.0 --position_dependent_phones false data/dict "<SPOKEN_NOISE>" data/local/ data/lang || exit 1;

# prepare language stuffs for test data
utils/prepare_lang.sh --sil-prob 0.0 --position_dependent_phones false data/dict_test "<SPOKEN_NOISE>" data/local_test/ data/lang_test || exit 1;

# monophone
steps/train_mono.sh --boost-silence 1.0 --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/mono || exit 1;

#monophone_ali train
steps/align_si.sh --boost-silence 1.0 --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/mono exp/mono_ali || exit 1;

word level alignment
steps/get_train_ctm.sh data/mfcc/train data/lang exp/mono_ali || exit 1;

for n in `seq $n`; do gunzip -k -f exp/mono_ali/ctm.$n.gz; done || exit 1;

 . ./ali2Phones.sh

#monophone_ali test
steps/align_si.sh --boost-silence 1.0 --nj $n --cmd "$train_cmd" data/mfcc/test data/lang_test exp/mono exp/mono_test_ali || exit 1;

# word level alignment
steps/get_train_ctm.sh data/mfcc/test data/lang_test exp/mono_test_ali || exit 1;

for n in `seq $n`; do gunzip -k -f exp/mono_test_ali/ctm.$n.gz; done || exit 1;

steps/get_prons.sh data/mfcc/test data/lang_test exp/mono_test_ali

gunzip -c exp/mono_test_ali/prons.*.gz | utils/sym2int.pl -f 4 data/lang_test/words.txt | utils/sym2int.pl -f 5- data/lang_test/phones.txt
