#!/bin/bash


input_folder=$1
output_folder=$2
missing_output_file=$3

num_dags=$(ls $input_folder | wc -l)

echo '' > $missing_dag_outputs
for i in `seq 0 $num_dags` ; 
do
    #if the dag number $i has not been computed
    if [ $(find "$output_folder" -name "schedule_dag_$i.json" | wc -l) -eq 0 ] ; 
    then
        echo $i >> $missing_dag_outputs
    fi
done