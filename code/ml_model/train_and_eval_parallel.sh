#!/bin/bash


for m in 6 7 8; do
  for n in 10 20 30; do
    for epo in {1..10}; do
      epochs=$((10 * epo));
      python3 ml_model.py $m $n $epochs &
    done
  done
done

wait

for n in 10 20; do
  for epo in {1..10}; do
    epochs=$((10 * epo));
    python3 ml_model.py 4 $n $epochs &
  done
done

wait

for epo in {1..10}; do
  epochs=$((10 * epo))
  python3 ml_model.py 2 10 $epochs &
done

wait

echo "finished evaluating all models !"