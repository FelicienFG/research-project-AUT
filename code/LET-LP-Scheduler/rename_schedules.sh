#!/bin/bash

data_folder="dag_tasks_output_schedules"

json_files="$(find $data_folder -type f -name "*.json")"
json_files=$(echo $json_files | xargs -n1 | sort -V | xargs)



rename_files() {
    sorted_files=$1
    id=0

    for file in $sorted_files ;
    do
        extension="${file##*.}";
        mv $file "$data_folder/schedule_dag_$id.$extension";
        id=$(($id+1))
    done
}

rename_files "$json_files"