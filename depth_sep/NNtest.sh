for N in 10 16 24 36 54 78 113 162 231
do
    for t in 0 # {0..9}
    do
        python experiments.py --dataset eldan --model NN --N ${N} --H 2 --trial ${t}
        wait
    done
    # python result_alloc.py --dataset eldan --model NN --N ${N} --n_epoch 100 --trials 10
    # wait
done
