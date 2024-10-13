#!/bin/bash


input_folder=$1
output_folder=$2
missing_output_file=$3

num_dags=$(ls $input_folder | wc -l)

echo -n '' > $missing_output_file
for i in `seq 0 $num_dags` ; 
do
    #if the dag number $i has not been computed
    if [ $(find "$output_folder" -name "schedule_dag_$i.json" | wc -l) -eq 0 ] ; 
    then
        echo $i >> "$missing_output_file"
    fi
done