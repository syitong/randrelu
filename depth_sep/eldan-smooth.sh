#!/bin/bash
# For the number of params to be the same with RF models,
# we set N to be -(d+1)/2 + sqrt((d+1)^2+4(d+1)M)
# if RF has M nodes and data is d dimensional.
for N in 20 40 80 160 320 640 1280 2560 5120
do
    # for trial in 0 1 2 3 4 5 6 7
    # do
    #   python experiments.py --action screen --N ${N} --trial ${trial} --file eldan-smooth-RF-params
    #   wait
    # done
    python result_alloc.py --action test --file eldan-smooth-RF-params --N ${N} --H 2
    wait
done

for N in 7 11 17 25 37 54 77 110 157
do
    # for trial in 0 1 2 3 4 5 6 7
    # do
    #   python experiments.py --action screen --N ${N} --trial ${trial} --file eldan-smooth-NN-params
    #   wait
    # done
    python result_alloc.py --action test --file eldan-smooth-NN-params --N ${N} --H 2
    wait
done
for N in 20 40 80 160 320 640 1280 2560 5120
do
    # for trial in 0 1 # 2 3 4 5 6 7
    # do
    #   python experiments.py --action screen --N ${N} --trial ${trial} --file eldan-smooth-RF-params
    #   wait
    # done
    python result_alloc.py --action test --file eldan-smooth-NN-params --N ${N} --H 1
    wait
done


# for N in 20 40 80 160 320 640 1280 2560 5120
# do
#     # for trial in 0 1 # 2 3 4 5 6 7 8 9
#     # do
#     #   python experiments.py --action test --N ${N} --trial ${trial} --file eldan-smooth-RF-params
#     #   wait
#     # done
#     python result_alloc.py --action screen --file eldan-smooth-RF-params --N ${N} --H 2
#     wait
# done
#
# for N in 7 # 11 17 25 37 54 77 110 157
# do
#     for trial in 0 # 1 2 3 4 5 6 7 8 9
#     do
#       python experiments.py --action test --N ${N} --trial ${trial} --file eldan-smooth-NN-params
#       wait
#     done
#     python result_alloc.py --action test --file eldan-smooth-NN-params  --N ${N} --H 2
#     wait
# done
# #
# for N in 20 40 80 160 320 640 1280 2560 5120
# do
#     # for trial in 0 1 # 2 3 4 5 6 7 8 9
#     # do
#     #   python experiments.py --action test --N ${N} --trial ${trial} --file eldan-smooth-RF-params
#     #   wait
#     # done
#     python result_alloc.py --action screen --file eldan-smooth-NN-params --N ${N} --H 1
#     wait
# done
