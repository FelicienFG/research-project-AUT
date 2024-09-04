#!/bin/bash

input_dir="dag_tasks_input_files"
output_dir="dag_tasks_output_schedules"

num_tasks=$(ls $input_dir | wc -l)
num_tasks=$(($num_tasks-1))
total_cores=8

task_step=$(($num_tasks/$total_cores))

compute_schedules_parallel()
{
    id_start=$1
    id_end=$2

    for dag_task_id in `seq $id_start $(($id_end-1))` ; 
    do
        echo "current start and current id: " $id_start $dag_task_id;
        python3 main.py --infile "$input_dir/multicore_system_DAGtask_$dag_task_id.json" --outfile "$output_dir/schedule_dag_$dag_task_id.json" --solver GUROBI ;
    done
}

for c in `seq 0 $(($total_cores-1))` ;
do

    start_task_id=$(($task_step*$c))
    end_task_id=$(($task_step*($c+1)))

    compute_schedules_parallel $start_task_id $end_task_id &
done