#!/bin/bash

input_dir="dag_tasks_input_files"
output_dir="dag_tasks_output_schedules"

for dag_task_id in {4..9} ; 
do
    python3 main.py --infile "$input_dir/multicore_system_DAGtask_$dag_task_id.json" --outfile "$output_dir/schedule_dag_$dag_task_id.json" --solver GUROBI ;
done