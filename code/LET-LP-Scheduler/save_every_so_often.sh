#!/bin/bash


i=0
while :
do
  #backup every half an hour
  sleep 1800;
  git add . ;
  git commit -m "model train and eval save number $i" ;
  git push ;
  i=$((i+1));
done