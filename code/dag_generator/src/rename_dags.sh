#!/bin/bash

data_folder="data"

gml_files="$(find $data_folder -type f -name "*.gml")"
gml_files=$(echo $gml_files | xargs -n1 | sort -V | xargs)

pickle_files="$(find $data_folder -type f -name "*.gpickle")"
pickle_files=$(echo $pickle_files | xargs -n1 | sort -V | xargs)

png_files="$(find $data_folder -type f -name "*.png")"
png_files=$(echo $png_files | xargs -n1 | sort -V | xargs)



rename_files() {
    sorted_files=$1
    id=0

    for file in $sorted_files ;
    do
        extension="${file##*.}";
        mv $file "$data_folder/Tau_$id.$extension";
        id=$(($id+1))
    done
}

rename_files "$gml_files"
rename_files "$pickle_files"
rename_files "$png_files"