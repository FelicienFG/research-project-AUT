#!/bin/bash

p=8
n_list="10 20 30 40 50"
m_list="2 4 6 7 8"

for m in $m_list ; 
do
  for n in $n_list ; 
  do
    data_file="data_p${p}n${n}";
    dag_file="dag_m${m}p${p}n${n}_input_files";
    python3 data_loader.py $data_file $dag_file $m &
  done
done
