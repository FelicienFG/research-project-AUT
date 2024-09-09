#!/bin/bash

input_dir="dag_tasks_input_files"
output_dir="dag_tasks_output_schedules"

dag_ids="112 113 114 115 116" 

for dag_task_id in $dag_ids ; 
do
    python3 main.py --infile "$input_dir/multicore_system_DAGtask_$dag_task_id.json" --outfile "$output_dir/schedule_dag_$dag_task_id.json" --solver GUROBI ;
done