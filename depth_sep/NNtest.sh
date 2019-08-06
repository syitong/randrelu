for N in 12 20 31 48 72 105 153 221 317
do
    for trial in 2 # {0..9}
    do
        python experiments.py --action screen --N ${N} --trial ${trial} --file daniely-NN-params --H 2
        wait
    done
    python result_alloc.py --action screen --file daniely-NN-params --N ${N} --H 2
    wait
done
