#!/bin/bash
# For the number of params to be the same with RF models,
# we set N to be -(d+1)/2 + sqrt((d+1)^2+4(d+1)M)
# if RF has M nodes and data is d dimensional.

for N in 7 11 17 25 37 54 77 110 157
do
    for trial in 0 1 2 3 4 5 6 7
    do
        python experiments.py --N ${N} --trial ${trial} --file eldan-NN-params
        wait
    done
    python result_alloc.py --file eldan-NN-params --N ${N}
    wait
done
