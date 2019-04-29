#!/bin/bash
# For the number of params to be the same with RF models,
# we set N to be -(d+1)/2 + sqrt((d+1)^2+4(d+1)M)
# if RF has M nodes and data is d dimensional.

for N in 10 16 24 36 54 78 113 162 231
do
    for trial in 0 1 2 3 4 5 6 7
    do
        python experiments.py --model NN --N ${N} --trial ${trial} --file eldan-params
        wait
    done
    python result_alloc.py --file eldan-params --N ${N}
    wait
done
