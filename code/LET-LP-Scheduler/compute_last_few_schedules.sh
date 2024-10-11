#!/bin/bash




p=8
n_list="10 20 30"
m_list="6 7 8"


for m in $m_list ;
do
    for n in $n_list ;
    do
        dag_ids="cat missing_file_ids_m${m}p${p}n${n}" 
        inputDir="dag_m${m}p${p}n${n}_input_files"
        outputDir="dag_m${m}p${p}n${n}_output_schedules"

        for dag_task_id in $dag_ids ; 
        do
                python3 main.py --infile "$inputDir/multicore_system_DAGtask_$dag_task_id.json" --outfile "$outputDir/schedule_dag_$dag_task_id.json" --solver GUROBI > /tmp/${outputDir}DAG_${dag_task_id}.log &
        done
    done
done