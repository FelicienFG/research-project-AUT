#!/bin/bash

p=8
m_list="2 4 6 7 8"
n_list="10 20 30"

for n in $n_list ; 
do
  count=$(ls ../dag_generator/data_p${p}n${n}) ;
  for m in $m_list ; 
  do
    for id in `seq 1 $count` ; do
      
    done
  done

done