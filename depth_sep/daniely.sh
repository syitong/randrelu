#!/bin/bash
# For the number of params to be the same with RF models,
# we set N to be -(d+1)/2 + sqrt((d+1)^2+4(d+1)M)
# if RF has M nodes and data is d dimensional.
for N in 20 40 # 80 160 320 640 1280 2560 5120
do
    for trial in 0 # 1 2 3 4 5 6 7
    do
      python experiments.py --action screen --N ${N} --trial ${trial} --file daniely-RF-params
      wait
    done
    python result_alloc.py --action screen --file daniely-RF-params --N ${N}
    wait
done

for N in 12 20 # 31 48 72 105 153 221 317
do
    for trial in 0 # 1 2 3 4 5 6 7
    do
      python experiments.py --action screen --N ${N} --trial ${trial} --file daniely-NN-params --H 2
      wait
    done
    python result_alloc.py --action screen --file daniely-NN-params --N ${N} --H 2
    wait
done
for N in 20 40 # 80 160 320 640 1280 2560 5120
do
    for trial in 0 # 1 2 3 4 5 6 7
    do
      python experiments.py --action screen --N ${N} --trial ${trial} --file daniely-RF-params --H 1
      wait
    done
    python result_alloc.py --action screen --file daniely-NN-params --N ${N} --H 1
    wait
done
