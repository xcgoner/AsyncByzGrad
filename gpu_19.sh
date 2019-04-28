#!/bin/bash

source /home/cx2/virtualenv/mxnet/bin/activate

export CUDA_VISIBLE_DEVICES=2

basedir=/home/cx2/datasets
logdir=/home/cx2/src/zeno_async/results

# training data
inputdir=$basedir/cifar10_zeno

watchfile=$logdir/gpu_19.log

# prepare the dataset
# python convert_cifar10_to_np_normalized_unbalanced.py --nsplit 100 --normalize 1 --step 8 --output $basedir/cifar10_unbalanced/
# python convert_cifar10_to_np_normalized.py --nsplit 100 --normalize 1 --output $basedir/cifar10_balanced/

model="default"
lr=0.1
method="zeno++"
byz="signflip"

for maxdelay in 20
do
    for nbyz in 4 8
    do
        for rho in 0.004
        do 
            for epsilon in 0 0.05 0.1 0.15
            do
                for zenodelay in 10
                do
                    logfile=$logdir/zenopp_tunezeno_${model}_${lr}_${method}_${maxdelay}_${byz}_${nbyz}_${rho}_${epsilon}_${zenodelay}.txt
                    > $logfile
                    for seed in 337 773 557 755 
                    do
                        python /home/cx2/src/zeno_async/zeno_async/train_cifar10_gpu.py --gpu 0 --classes 10 --model ${model} --nworkers 10 --nbyz ${nbyz} --byz-type ${byz} --byz-test ${method} --rho ${rho} --epsilon ${epsilon} --zeno-delay ${zenodelay} --batchsize 128 --lr ${lr} --lr-decay 0.1 --lr-decay-epoch 100,150 --epochs 200 --seed ${seed} --max-delay ${maxdelay} --dir $inputdir --log $logfile 2>&1 | tee $watchfile
                    done
                done
            done
        done
    done
done