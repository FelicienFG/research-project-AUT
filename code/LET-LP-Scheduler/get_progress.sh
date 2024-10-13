#!/bin/bash


p=8
n_list="10 20 30 40 50"
m_list="2 4 6 7 8"
progress_file="progress.progress"

echo -n '' > $progress_file

for m in $m_list ;
do
    for n in $n_list ; 
    do
      if [[ m -eq 2 && ( $n -eq 30 || $n -eq 40 || $n -eq 50 ) ]] ; then
        echo "don't do anything"
      else
        in_files=$(ls "dag_m${m}p${p}n${n}_input_files" | wc -l)
        out_files=$(ls "dag_m${m}p${p}n${n}_output_schedules" | wc -l)
        echo "${out_files}/${in_files}" >> $progress_file
      fi
    done
done