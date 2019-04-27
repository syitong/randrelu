#!/bin/bash

for N in 20
do
    for trial in 0 1
    do
        python experiments.py --N ${N} --trial ${trial} --file eldan-params
    wait
    done
    python result_alloc.py --file eldan-params --N ${N}
    wait
    python result_show.py --dataset eldan --N 20
done
