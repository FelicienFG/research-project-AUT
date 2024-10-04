#!/bin/bash

p=8
n_list="10 20 30 40 50"
m_list="2 4 6 7 8"

for m in $m_list ;
do
    for n in $n_list ; 
    do
        ./compute_missing_files.sh "dag_m${m}p${p}n${n}_input_files" "dag_m${m}${p}${n}_output_schedules" "missing_file_ids_m${m}p${p}n${n}" &
    done
done