#!/bin/bash

data_folder="dag_tasks_output_schedules"
ids_to_remove=$(cat schedules_to_remove)

for id in $ids_to_remove ;
do
    rm $data_folder/schedule_dag_$id.json
done