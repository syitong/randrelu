for N in 20 40 80 160 320 640 1280 2560 5120
do
    # for t in 0 1 2 3 4 5 6 7 8 9
    # do
    #     python experiments.py --dataset eldan --model RF --N ${N} --H 1 --trial ${t}
    #     wait
    # done
    python result_alloc.py --dataset eldan --model RF --N ${N} --n_epoch 100 --trials 1
    wait
done
