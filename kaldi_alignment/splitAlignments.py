#!/bin/sh

#  splitAlignments.py
#  
#
#  Created by Eleanor Chodroff on 3/25/15.
#
#
#
import sys
import csv
import os

results=[]

# name = name of first text file in final_ali.txt
# name_fin = name of final text file in final_ali.txt

name = "Dan_jingju_a_cappella_singing_dataset_danAll_daeh-Yang_Yu_huan-Tai_zhen_wai_zhuan-lon"
name_fin = "Laosheng_primary_school_recording_20171217TianHao_lsxp-Wei_guo_jia-Hong_yang_dong-sizhu_teacher"

try:
    with open("./exp/mono_ali/final_ali.txt") as f:
        next(f) #skip header
        for line in f:
            columns=line.split("\t")
            name_prev = name
            dataset = name_prev.split('_')[0]
            name = columns[1]
            if (name_prev != name):
                try:
                    path_roletype = os.path.join('./splitAli',dataset.lower())
                    if not os.path.exists(path_roletype):
                        os.makedirs(path_roletype)
                    with open(os.path.join('./splitAli',dataset.lower(),name_prev[len(dataset)+1:]+".txt"),'w') as fwrite:
                        writer = csv.writer(fwrite)
                        fwrite.write("\n".join(results))
                        fwrite.close()
                #print name
                except Exception as e:
                    print("Failed to write file", e)
                    sys.exit(2)
                del results[:]
                results.append(line[0:-1])
            else:
                results.append(line[0:-1])
except Exception as e:
    print("Failed to read file", e)
    sys.exit(1)
# this prints out the last textfile (nothing following it to compare with)
try:
    with open(os.path.join('./splitAli', dataset.lower(), name_prev[len(dataset)+1:]+".txt"),'w') as fwrite:
        writer = csv.writer(fwrite)
        fwrite.write("\n".join(results))
        fwrite.close()
                #print name
except Exception as e:
    print("Failed to write file", e)
    sys.exit(2)