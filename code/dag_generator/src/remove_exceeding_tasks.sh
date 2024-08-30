#!/bin/bash

data_folder="data"
ids_to_remove=$(cat tasks_to_remove)

for id in $ids_to_remove ;
do
    rm $data_folder/Tau_$id.*
done