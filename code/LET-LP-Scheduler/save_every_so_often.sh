#!/bin/bash


i=0
while :
do
  #backup every half an hour
  sleep 900;
  git add . ;
  git commit -m "ILP schedules computation save number $i" ;
  git push ;
  i=$((i+1));
done