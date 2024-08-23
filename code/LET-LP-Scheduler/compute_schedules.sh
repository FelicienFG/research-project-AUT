#!/bin/bash

input_dir="dag_tasks_input_files"
output_dir="dag_tasks_output_schedules"

dag_task_id=0
for file in $input_dir/* ; 
do
    python3 main.py --infile "$file" --outfile "$output_dir/schedule_dag_$dag_task_id.json" --solver PULP_CBC_CMD ;
    dag_task_id=$(($dag_task_id+1))
done