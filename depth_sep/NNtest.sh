for N in 7 11 17 25 37 54 77 110 157
do
    # for t in 1 # {0..9}
    # do
    #     python experiments.py --dataset eldan --model NN --N ${N} --H 2 --trial ${t}
    #     wait
    # done
    python result_alloc.py --dataset eldan --model NN --N ${N} --n_epoch 100 --trials 10
    wait
done
