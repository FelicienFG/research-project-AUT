#!/bin/bash

p=$1
n=$2
data_file="data_p${p}n${n}/"

echo $data_file

python3 src/daggen-cli.py ;#generate data folder

mv "data" $data_file ;#move to the correctly named one

echo "generated data folder, proceeding to isolate removable tasks..."
python3 src/utility.py $n $data_file ; #isolate the tasks needing removal

echo "finished isolating removable tasks, proceeding to removing them..."
./src/remove_exceeding_tasks.sh $data_file ;

echo "finished removing the isolated tasks, proceeding to renaming them..."
./src/rename_dags.sh $data_file ;

echo "finished generating the dataset."
exit 0