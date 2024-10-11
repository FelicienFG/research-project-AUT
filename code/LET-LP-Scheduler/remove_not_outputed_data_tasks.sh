#!/bin/bash

p=8
#m_list="2 4 6 7 8"
n_list="10 20"
m=2

for n in $n_list ; 
do
  for id in $(cat missing_file_ids_m${m}p${p}n${n}) ; do
    rm ../dag_generator/data_p${p}n${n}/Tau_${id}.*
  done
done