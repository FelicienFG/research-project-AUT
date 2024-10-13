#!/bin/bash

data_folder=$1
ids_to_remove=$(cat tasks_to_remove)

echo "$ids_to_remove" | xargs -I {} -P 90 bash -c 'rm '"$data_folder"'/Tau_{}.*'