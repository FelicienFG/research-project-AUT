#!/bin/bash




p=8
n_list="20"
m_list="6"


for m in $m_list ; 
do
  for n in $n_list ;
  do
    dag_ids="$(cat ../ml_model/null_jsons_m${m}p${p}n${n})" 
    inputDir="dag_m${m}p${p}n${n}_input_files"
    outputDir="dag_m${m}p${p}n${n}_output_schedules"
    for dag_task_id in $dag_ids ; 
    do
            python3 main_ilp.py --infile "$inputDir/multicore_system_DAGtask_$dag_task_id.json" --outfile "$outputDir/schedule_dag_$dag_task_id.json" --solver GUROBI > /tmp/${outputDir}DAG_${dag_task_id}.log &
    done
  done
done