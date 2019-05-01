for N in 20 40 80 160 320 640 1280 2560 5120
do
    # for t in 5 7 # {0..9}
    # do
    #     python experiments.py --dataset eldan --model NN --N ${N} --H 1 --trial ${t}
    #     wait
    # done
    python result_alloc.py --dataset eldan --model NN --N ${N} --n_epoch 100 --trials 10
    wait
done
