# for N in 7 11 17 25 37 54 77 110 157
for N in 20 40 80 160 320 640 1280 2560 5120
do
    for trial in {0..7}
    do
        python experiments.py --action screen --N ${N} --trial ${trial} --file daniely-NN-params --H 1
        wait
    done
    python result_alloc.py --action screen --file daniely-NN-params --N ${N} --H 1
    wait
done
