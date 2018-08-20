#!/usr/bin/env bash

#for i in exp/mono_ali/ali.*.gz; do
#	src/bin/ali-to-phones --ctm-output exp/mono_ali/final.mdl ark:"gunzip -c $i|" -> ${i%.gz}.ctm;
#done;
#
#cd exp/mono_ali
#
#cat *.ctm > merged_alignment.txt
#
#cd ../..
#
#R -f id2phone.R
#
#python splitAlignments.py
#
#cd ~/Desktop
#rm -rf tmp && mkdir tmp || exit 1;
#
#header="/Users/ronggong/PycharmProjects/mispronunciation-detection/kaldi_alignment/header.txt"
#
## direct the terminal to the directory with the newly split session files
## ensure that the RegEx below will capture only the session files
## otherwise change this or move the other .txt files to a different folder
#
#for x in laosheng dan;do
#	cd /Users/ronggong/PycharmProjects/mispronunciation-detection/kaldi_alignment/splitAli/$x
#	for i in *.txt; do
#		cat "$header" "$i" > /Users/ronggong/Desktop/tmp/xx.$$
#		mv /Users/ronggong/Desktop/tmp/xx.$$ "$i"
#	done
#done
#	cd ../..
#
#/Applications/Praat.app/Contents/MacOS/Praat "createtextgridDan.praat"
/Applications/Praat.app/Contents/MacOS/praat "createtextgridLaosheng.praat"