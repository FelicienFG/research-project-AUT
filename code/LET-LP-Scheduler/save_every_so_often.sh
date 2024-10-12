#!/bin/bash


i=0
while :
do
  #backup every 45 min
  sleep 2700;
  git add . ;
  git commit -m "model train and eval save number $i" ;
  git push ;
  i=$((i+1));
done