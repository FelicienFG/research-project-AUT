#!/bin/bash

p=8
n_list="30 40 50"
m_list="2"
total_cores=20

compute_schedules_parallel()
{
    id_start=$1
    id_end=$2
    inputDir=$3
    outputDir=$4

    for dag_task_id in `seq $id_start $(($id_end-1))` ; 
    do
        #echo "current start and current id: " $id_start $dag_task_id;
        python3 main_ilp.py --infile "$inputDir/multicore_system_DAGtask_$dag_task_id.json" --outfile "$outputDir/schedule_dag_$dag_task_id.json" --solver GUROBI > /tmp/${outputDir}DAG_${dag_task_id}.log;
    done
}

for m in $m_list ;
do
  for n in $n_list ;
  do
    input_dir="dag_m${m}p${p}n${n}_input_files"
    output_dir="dag_m${m}p${p}n${n}_output_schedules"
    num_tasks=$(ls $input_dir | wc -l)
    num_tasks=$(($num_tasks-1))

    task_step=$(($num_tasks/$total_cores))

    for c in `seq 0 $(($total_cores-1))` ;
    do

        start_task_id=$(($task_step*$c))
        end_task_id=$(($task_step*($c+1)))

        compute_schedules_parallel $start_task_id $end_task_id $input_dir $output_dir &
    done

  done
done