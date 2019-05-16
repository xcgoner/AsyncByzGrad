#!/bin/bash
#PBS -l select=11:ncpus=112 -lplace=excl

source /opt/intel/compilers_and_libraries_2017/linux/mpi/bin64/mpivars.sh
source activate mxnet_latest


### OPA FABRIC ###
export I_MPI_FABRICS=ofi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=56


### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

export KMP_AFFINITY=granularity=fine,compact,1,0;

basedir=/homes/cx2/datasets
logdir=/homes/cx2/zeno_async/results

# training data
inputdir=$basedir/cifar10_zeno

watchfile=$logdir/exp_52.log

# prepare the dataset
# python convert_cifar10_to_np_normalized_unbalanced.py --nsplit 100 --normalize 1 --step 8 --output $basedir/cifar10_unbalanced/
# python convert_cifar10_to_np_normalized.py --nsplit 100 --normalize 1 --output $basedir/cifar10_balanced/

model="default"
lr=0.1
method="zeno++"
byz="signflip"
rho=0.001
epsilon=0

for maxdelay in 15
do
    for nbyz in 4 8
    do
        for zenobatchsize in 16 96
        do
            logfile=$logdir/zenopp_tunebatchsize_${model}_${lr}_${method}_${maxdelay}_${byz}_${nbyz}_${rho}_${epsilon}_${zenobatchsize}.txt
            > $logfile
            for seed in 337 773 557 755 
            do
                python /homes/cx2/zeno_async/zeno_async/train_cifar10.py --classes 10 --model ${model} --nworkers 10 --nbyz ${nbyz} --byz-type ${byz} --byz-test ${method} --rho ${rho} --epsilon ${epsilon} --zeno-delay 10 --zeno-batchsize ${zenobatchsize} --batchsize 128 --lr ${lr} --lr-decay 0.1 --lr-decay-epoch 100,150 --epochs 200 --seed ${seed} --max-delay ${maxdelay} --dir $inputdir --log $logfile 2>&1 | tee $watchfile
            done
        done
    done
done
