#!/bin/bash

for N in 20 40 80 160 320 640 1280 2560 5120
do
    for trial in 0 1 2 3 4 5 6
    do
        python experiments.py --N ${N} --trial ${trial} --file eldan-params
        wait
    done
    python result_alloc.py --file eldan-params --N ${N}
    wait
done
