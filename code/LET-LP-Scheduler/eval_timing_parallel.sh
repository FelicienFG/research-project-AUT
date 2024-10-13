#!/bin/bash


#m = 2
for n in 10 20 ;
do
  python3 eval_timing_ILP.py 2 8 $n > /tmp/m2p8n${n}_timeres.log &
done

#m = 4
for n in 10 20 30 40 50 ;
do
  python3 eval_timing_ILP.py 4 8 $n > /tmp/m4p8n${n}_timeres.log &
done

#other m
for m in 6 7 8 ;
do
  for n in 10 20 30 40 50 ;
  do
    python3 eval_timing_ILP.py $m 8 $n > /tmp/m${m}p8n${n}_timeres.log &
  done
done
