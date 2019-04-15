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

watchfile=$logdir/experiment_script_3.log

logfile=$logdir/experiment_script_3.txt

# prepare the dataset
# python convert_cifar10_to_np_normalized_unbalanced.py --nsplit 100 --normalize 1 --step 8 --output $basedir/cifar10_unbalanced/
# python convert_cifar10_to_np_normalized.py --nsplit 100 --normalize 1 --output $basedir/cifar10_balanced/

> $logfile

# start training
# python /homes/cx2/zeno_async/zeno_async/train_cifar10.py --classes 10 --model default --nworkers 10 --nbyz 3 --byz-type signflip --byz-test zeno++ --rho 0.001 --epsilon 0 --zeno-delay 5 --batchsize 128 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --epochs 200 --seed 337 --max-delay 10 --dir $inputdir --log $logfile 2>&1 | tee $watchfile
python /homes/cx2/zeno_async/zeno_async/train_cifar10.py --classes 10 --model default --nworkers 10 --nbyz 6 --byz-type signflip --byz-test zeno++ --rho 0.001 --epsilon 0 --zeno-delay 10 --batchsize 128 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --epochs 200 --seed 337 --max-delay 10 --dir $inputdir --log $logfile 2>&1 | tee $watchfile
