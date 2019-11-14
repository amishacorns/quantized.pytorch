#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate p37t1
#@ is used to represent a None like value that ignores the parameter (ie not used). return second argument as prefix
function getarg() {
if [ $1 == '@' ];
then echo '';
else echo $2 $1;
fi; }

function arr_to_str_delim() {
tmp=$0
for i in ${0[@]:1}; do tmp="$tmp$1$i"; done
echo $tmp
 }
RUNNING_PROCS=()
#set -x
#set -e
#general settings
#script accepts experiment ÍD argument
WRN28_10_C100_PATH=results/wresnet28-10_ba_m10_cifar100/checkpoint.pth.tar
R44_C100_PATH=results/resnet44_ba_m40_cifar100/model_best.pth.tar
R44_C10_PATH=results/resnet44_cifar10/model_best.pth.tar
R18_IMGNT_PATH=/home/mharoush/.torch/models/resnet18-5c106cde.pth
##### default settings
#STEP_LIMIT="--steps-limit 16000"
DEVICE_ID_C=(0 1 2 3)
EXP_RUNNER_LOGDIR='exp_runner_logs'
N_EXP_PER_GPU=1
EXP=${1:-1}
DATASET_C=('cifar10-raw')
DATA_SIZE_C=('@')
AUX_C=('smoothl1')
BN_MOD_C=('@')
LOSS_C=('kld')
LR_C=(100.)
MIXUP_C=('--mixup')
SEED_C=(123)
AUX_SCALE_C=(0.01)
BATCH_SIZE=512
W_BITS=2
ACT_BITS=4
CV_FC_A=4
CV_FC_W=4
STEPS_PER_EPOCH=200
W_RANGE_MOD='--free-w-range'
T_W_PATH=$R44_C10_PATH
EXT_MODEL_CFG=
DEPTH=44
##### BEGIN
if [ ! -d $EXP_RUNNER_LOGDIR ]
then mkdir -p $EXP_RUNNER_LOGDIR
fi

x=$DEVICE_ID_C
for i in ${DEVICE_ID_C[@]:1}; do x=$x,$i; done
export CUDA_VISIBLE_DEVICES=$x

#if [ $EXP -eq 1 ]
#then
#    ##### EXP=1 - find best lr per loss function
#    EXP_G='loss_and_lr'
#    LOSS_C=('mse' )
#    #LR_C=(50 5 2) loss diverged under these values
#    # best 90.9 0.6951928
#    LR_C=(0.6951928)
#    #####
#elif [ $EXP -eq 11 ]
#then
#    ##### EXP=1 - find best lr per loss function
#    EXP_G='loss_and_lr'
#    LOSS_C=('smoothl1')
#    #LR_C=(50 5 ) loss diverged under these values
#    # best 90.13 0.6951928
#    #####
#elif [ $EXP -eq 12 ]
#then
#    ##### EXP=1 - find best lr per loss function
#    EXP_G='loss_and_lr'
#    LOSS_C=('kld')
#    LR_C=(100 70 30) #best 91.24 50
#    #####
#
#elif [ $EXP -eq 2 ]
#then
#    ##### EXP=2 -compare results of different batch normalization schemes
#    EXP_G='loss_and_lr'
#    BN_MOD_C=('--fresh-bn') #,'--absorb-bn')
#    LOSS_C=('mse')
#    LR_C=(0.6951928)
#    #####
#elif [ $EXP -eq 21 ]
#then
#    ##### EXP=2 -compare results of different batch normalization schemes
#    EXP_G='loss_and_lr'
#    BN_MOD_C=('--fresh-bn') #,'--absorb-bn')
#    LOSS_C=('smoothl1')
#    LR_C=(1.)
#    #####
#elif [ $EXP -eq 22 ]
#then
#    ##### EXP=2 -compare results of different batch normalization schemes
#    EXP_G='loss_and_lr'
#    BN_MOD_C=('--fresh-bn') #,'--absorb-bn')
#    LOSS_C=('kld')
#    LR_C=(100)
#    #####
#elif [ $EXP -eq 3 ]
#then
#    ##### EXP=3 - compare aux losses + mixup
#    EXP_G='aux_and_mixup'
#    BN_MOD_C=('@','--fresh-bn') #todo use best from exp 2
#    AUX_C=('mse' 'smoothl1' 'cos') #todo search over loss/aux scales?
#    LOSS_C=('mse')
#    MIXUP_C=('@' '--mixup') #todo test on main loss only
#    LR_C=(0.7)
#    #####
#elif [ $EXP -eq 301 ]
#then
#    ##### EXP=3 - compare aux losses + mixup
#    EXP_G='aux_and_mixup'
#    BN_MOD_C=('@','--fresh-bn') #todo use best from exp 2
#    AUX_C=('kld') #todo search over loss/aux scales?
#    LOSS_C=('mse')
#    MIXUP_C=('@' '--mixup') #todo test on main loss only
#    LR_C=(0.7)
#    #####
#elif [ $EXP -eq 31 ]
#then
#    ##### EXP=3 - compare aux losses + mixup
#    EXP_G='aux_and_mixup'
#    BN_MOD_C=('@') #todo use best from exp 2
#    AUX_C=('kld' ) #todo search over loss/aux scales?
#    AUX_SCALE_C=(0.6 0.3 0.06 0.01 0.005)
#    LOSS_C=('kld')
#    MIXUP_C=('@' '--mixup')
#    LR_C=(100)
#    #####
#elif [ $EXP -eq 311 ]
#then
#    ##### EXP=3 - compare aux losses + mixup
#    EXP_G='aux_and_mixup'
#    AUX_C=('mse' 'smoothl1' 'cos')
#    LOSS_C=('kld')
#    AUX_SCALE_C=(0.01 0.005) #mse tried 0.1, 0.05
#    MIXUP_C=('@' '--mixup')
#    LR_C=(100)
#    #####
if [ $EXP -eq 1 ]
then
    ##### EXP=1 - cifar10 dataset alternatives
    DATASET_C=('random-cifar10' 'imagine-cifar10-r44-no_dd_kl_r1000' 'imagine-cifar10-r44-dd-exp_kl_r1000_k5s1' \
                'imagine-cifar10-r44-dd-exp_r1000_k5s1')
    EXP_G='abs@cifar10_datasets_final'
    BN_MOD_C=('@') #todo best from exp 2 --fresh-bn
    DATA_SIZE_C=(50 4000)
    AUX_C=('smoothl1')
    AUX_SCALE_C=(0.01) #todo best from exp3
    LOSS_C=('kld') #todo best from exp3
    LR_C=(100)
    MIXUP_C=('--mixup')
    BATCH_SIZE=512
    W_BITS=2
    ACT_BITS=4
    CV_FC_A=4
    CV_FC_W=4
    SEED_C=(1)
    N_EXP_PER_GPU=2
#elif [ $EXP -eq 11 ]
#then
#    ##### EXP=1 cifar10 dataset alternatives
#    DATASET_C=('random-cifar10' 'imagine-cifar10-r44-no_dd_kl_r1000' 'imagine-cifar10-r44-dd-exp_kl_r1000_k5s1' \
#                'imagine-cifar10-r44-dd-exp_r1000_k5s1')
#    EXP_G='abs@cifar10'
#    BN_MOD_C=('@')
#    DATA_SIZE_C=(4000 50)
#    AUX_C=('smoothl1')
#    AUX_SCALE_C=(0.01) #todo best from exp3
#    LOSS_C=('kld') #todo best from exp3
#    LR_C=(100)
#    MIXUP_C=('--mixup')
#    BATCH_SIZE=512
#    W_BITS=2
#    ACT_BITS=4
#    CV_FC_A=4
#    CV_FC_W=4
#    SEED_C=(1)
#    N_EXP_PER_GPU=2
elif [ $EXP -eq 11 ]
then
    ##### EXP=1 - data impact raw
    DATASET_C=('cifar10-raw' 'imagine-cifar10-r44-no_dd_kl_r1000' 'imagine-cifar10-r44-no_dd_kl_r500' \
                'imagine-cifar10-r44-no_dd_kl_r100' 'imagine-cifar10-r44-no_dd_kl_r50' \
                'imagine-cifar10-r44-no_dd_kl_r20'  'imagine-cifar10-r44-no_dd_kl_r10')
    EXP_G='abs@cifar10_data_size_final'
    EXT_FLAGS="--calibration-set-size 5 --shuffle-calibration-steps 200"
    BN_MOD_C=('@') #todo best from exp 2 --fresh-bn
    DATA_SIZE_C=(50 100 500 1000 2000 4000 1 10)
    AUX_C=('smoothl1')
    AUX_SCALE_C=(0.01) #todo best from exp3
    LOSS_C=('kld') #todo best from exp3
    LR_C=(100)
    MIXUP_C=('--mixup')
    BATCH_SIZE=512
    W_BITS=2
    ACT_BITS=4
    CV_FC_A=4
    CV_FC_W=4
    SEED_C=(2 3 4 5 6)
    N_EXP_PER_GPU=1

elif [ $EXP -eq 2 ]
then
    EXP_G='abs@wr28-10-cifar100_final'
    #'cifar10-raw' 'imagine-cifar10-r44-r1000' 'imagine-cifar10-r44-r5000' 'imagine-cifar10-r44-r10000' 'imagine-cifar10-r44-r15000' 'imagine-cifar10-r44-r20000'
    DATASET_C=('cifar100-raw' 'random-cifar100' 'imagine-cifar100-wr28-10-no_dd_kl_r1000' \
                'imagine-cifar100-wr28-10-dd-exp_kl_r1000_k5s0.375' \
                'imagine-cifar100-wr28-10-dd-exp_r1000_k5s0.375')
    EXT_FLAGS="--calibration-set-size 50 --shuffle-calibration-steps 200" #"--recalibrate --calibration-set-size 5 --shuffle-calibration-steps 200"
    BATCH_SIZE=256
    DEPTH=28
    DATA_SIZE_C=(50 200)
    AUX_C=('smoothl1')
    AUX_SCALE_C=(0.01) #todo best from exp3
    LOSS_C=('kld') #todo best from exp3
    LR_C=(100)
    MIXUP_C=('--mixup')
    SEED_C=(1)
    T_W_PATH=$WRN28_10_C100_PATH
    EXT_MODEL_CFG=",'width':[160,320,640]"
    N_EXP_PER_GPU=1
    W_BITS=4
    ACT_BITS=4
    CV_FC_A=4
    CV_FC_W=4
elif [ $EXP -eq 3 ]
then
    EXP_G='abs@r18-imagenet_final'
    N_EXP_PER_GPU=1
    DATASET_C=('imagenet' 'random-imagenet' 'imagine-imagenet-r18-no_dd_kl_r1000' 'imagine-imagenet-r18_dd-exp_kl_r1000_k5s1' 'imagine-imagenet-r18_dd-exp_r1000_k5s1')
    EXT_FLAGS=  #"--recalibrate --calibration-set-size 10 --shuffle-calibration-steps 200"
    T_W_PATH=$R18_IMGNT_PATH
    AUX_C=('smoothl1')
    AUX_SCALE_C=(0.01) #todo best from exp3
    MIXUP_C=('--mixup')
    LR_C=(999)
    DATA_SIZE_C=(10)
    STEPS_PER_EPOCH=400
    DEPTH=18
    BATCH_SIZE=128
    W_BITS=4
    ACT_BITS=4
    CV_FC_A=8
    CV_FC_W=8
    SEED_C=(1)
elif [ $EXP -eq 4 ]
then
    ##### EXP=4 - compare data impact mixup
    DATASET_C=('imagine-cifar10-r44-no_dd_r10' 'imagine-cifar10-r44-no_dd_r50' 'imagine-cifar10-r44-no_dd_r100' 'imagine-cifar10-r44-no_dd_r500' 'imagine-cifar10-r44-no_dd_r1000' )
    EXP_G='data_size_new'
    BN_MOD_C=('@') #todo best from exp 2 --fresh-bn
    DATA_SIZE_C=(1 10 50 100 500 1000 2000 3000 4000)
    AUX_C=('smoothl1')
    AUX_SCALE_C=(0.01) #todo best from exp3
    LOSS_C=('kld') #todo best from exp3
    LR_C=(100)
    MIXUP_C=('--mixup') #mixup is generally better
    SEED_C=(1 2 3 4 5)
    #####
elif [ $EXP -eq 41 ]
then
    ##### EXP=4 - compare data impact mixup
    # subset for kld aux loss
    EXP_G='data_size'
    BN_MOD_C=('@')
    DATA_SIZE_C=(1 10 50 100 500 1000 2000)
    AUX_C=('kld')
    AUX_SCALE_C=(0.05 0.01 0.005) #0.05 excluded
    LOSS_C=('kld')
    LR_C=(100)
    MIXUP_C=('@' '--mixup')
    #####
elif [ $EXP -eq 411 ]
then
    ##### EXP=4 - compare data impact mixup
    #subset no aux loss
    EXP_G='data_size'
    BN_MOD_C=('@' '--fresh-bn')
    DATA_SIZE_C=(1 10 50 100 500 1000 2000)
    AUX_C=('@')
    #AUX_SCALE_C=(0.05 0.01 0.005) #0.05 excluded
    LOSS_C=('kld')
    LR_C=(100)
    MIXUP_C=('@' '--mixup')
    #####
elif [ $EXP -eq 5 ]
then
    EXP_G='abs@quality_calibration_r44-cifar10_2W4A'
    #'cifar10-raw' 'imagine-cifar10-r44-r1000' 'imagine-cifar10-r44-r5000' 'imagine-cifar10-r44-r10000' 'imagine-cifar10-r44-r15000' 'imagine-cifar10-r44-r20000'
    DATASET_C=('imagine-cifar10-r44-dd-exp_r1000_k5s1' 'imagine-cifar10-r44-dd-exp_kl_r1000_k5s1' 'cifar10-raw' 'random-cifar10' 'imagine-cifar10-r44-no_dd_sym_r1000')    #'imagine-cifar10-r44-dd-exp_kl_r1000_k5s1' 'imagine-cifar10-r44-dd-exp_kl_r1000' 'imagine-cifar10-r44-dd-exp_r1000' 'cifar10-raw' 'random-cifar10' 'imagine-cifar10-r44-dd-exp_r1000' 'imagine-cifar10-r44-no_dd_sym_r1000' 'imagine-cifar10-r44-dd-exp_kl_r1000' )
    # 'imagine-cifar10-r44-dd-exp_r1000' 'cifar10-raw' 'random-cifar10' 'imagine-cifar10-r44-dd-exp_r1000' 'imagine-cifar10-r44-no_dd_sym_r1000' 'imagine-cifar10-r44-dd-exp_kl_r1000' )
    EXT_FLAGS="--recalibrate --calibration-set-size 50 --shuffle-calibration-steps 200"
    BATCH_SIZE=256
    W_BITS=2
    ACT_BITS=4
    CV_FC_A=4
    CV_FC_W=4
    SEED_C=(1 2 3 4 5)
    N_EXP_PER_GPU=3
    MIXUP_C=('@')

elif [ $EXP -eq 501 ]
then
    EXP_G='abs@quality_calibration_r44-cifar10'
    #'cifar10-raw' 'imagine-cifar10-r44-r1000' 'imagine-cifar10-r44-r5000' 'imagine-cifar10-r44-r10000' 'imagine-cifar10-r44-r15000' 'imagine-cifar10-r44-r20000'
    DATASET_C=('imagine-cifar10-r44-dd-exp_r1000_k5s1' 'imagine-cifar10-r44-dd-exp_kl_r1000_k5s1' 'cifar10-raw' 'random-cifar10' 'imagine-cifar10-r44-no_dd_sym_r1000')    #'imagine-cifar10-r44-dd-exp_kl_r1000_k5s1' 'imagine-cifar10-r44-dd-exp_kl_r1000' 'imagine-cifar10-r44-dd-exp_r1000' 'cifar10-raw' 'random-cifar10' 'imagine-cifar10-r44-dd-exp_r1000' 'imagine-cifar10-r44-no_dd_sym_r1000' 'imagine-cifar10-r44-dd-exp_kl_r1000' )
    #'imagine-cifar10-r44-dd-exp_kl_r1000_k5s1' 'imagine-cifar10-r44-dd-exp_kl_r1000' 'imagine-cifar10-r44-dd-exp_r1000' 'cifar10-raw' 'random-cifar10' 'imagine-cifar10-r44-dd-exp_r1000' 'imagine-cifar10-r44-no_dd_sym_r1000' 'imagine-cifar10-r44-dd-exp_kl_r1000' )
    EXT_FLAGS="--recalibrate --calibration-set-size 50 --shuffle-calibration-steps 200"
    BATCH_SIZE=256
    W_BITS=4
    ACT_BITS=4
    CV_FC_A=8
    CV_FC_W=8
    SEED_C=(1 2 3 4 5)
    N_EXP_PER_GPU=3
    MIXUP_C=('@')
elif [ $EXP -eq 502 ]
then
    EXP_G='abs@quality_calibration_r44-cifar10'
    #'cifar10-raw' 'imagine-cifar10-r44-r1000' 'imagine-cifar10-r44-r5000' 'imagine-cifar10-r44-r10000' 'imagine-cifar10-r44-r15000' 'imagine-cifar10-r44-r20000'
    DATASET_C=('imagine-cifar10-r44-dd-exp_r1000_k5s1' 'imagine-cifar10-r44-dd-exp_kl_r1000_k5s1' 'cifar10-raw' 'random-cifar10' 'imagine-cifar10-r44-no_dd_sym_r1000')    #'imagine-cifar10-r44-dd-exp_kl_r1000_k5s1' 'imagine-cifar10-r44-dd-exp_kl_r1000' 'imagine-cifar10-r44-dd-exp_r1000' 'cifar10-raw' 'random-cifar10' 'imagine-cifar10-r44-dd-exp_r1000' 'imagine-cifar10-r44-no_dd_sym_r1000' 'imagine-cifar10-r44-dd-exp_kl_r1000' )
    EXT_FLAGS="--recalibrate --calibration-set-size 50 --shuffle-calibration-steps 200"
    BATCH_SIZE=256
    W_BITS=8
    ACT_BITS=4
    CV_FC_A=8
    CV_FC_W=8
    SEED_C=(1 2 3 4 5)
    N_EXP_PER_GPU=3
    MIXUP_C=('@')
elif [ $EXP -eq 503 ]
then
    EXP_G='abs@quality_calibration_r44-cifar10'
    #'cifar10-raw' 'imagine-cifar10-r44-r1000' 'imagine-cifar10-r44-r5000' 'imagine-cifar10-r44-r10000' 'imagine-cifar10-r44-r15000' 'imagine-cifar10-r44-r20000'
    DATASET_C=('imagine-cifar10-r44-dd-exp_r1000_k5s1' 'imagine-cifar10-r44-dd-exp_kl_r1000_k5s1' 'cifar10-raw' 'random-cifar10' 'imagine-cifar10-r44-no_dd_sym_r1000')
    EXT_FLAGS="--recalibrate --calibration-set-size 50 --shuffle-calibration-steps 200"
    BATCH_SIZE=256
    W_BITS=4
    ACT_BITS=8
    CV_FC_A=8
    CV_FC_W=8
    SEED_C=(1 2 3 4 5)
    N_EXP_PER_GPU=3
    MIXUP_C=('@')
elif [ $EXP -eq 51 ]
then
    EXP_G='abs@quality_calibration_wr28-10-cifar100'
    #'cifar10-raw' 'imagine-cifar10-r44-r1000' 'imagine-cifar10-r44-r5000' 'imagine-cifar10-r44-r10000' 'imagine-cifar10-r44-r15000' 'imagine-cifar10-r44-r20000'
    DATASET_C=('cifar100-raw' 'random-cifar100' 'imagine-cifar100-wr28-10-no_dd_kl_r1000' 'imagine-cifar100-wr28-10-dd-exp_kl_r1000_k5s0.375' 'imagine-cifar100-wr28-10-dd-exp_r1000_k5s0.375')
    EXT_FLAGS="--recalibrate --calibration-set-size 5 --shuffle-calibration-steps 200"
    BATCH_SIZE=256
    DEPTH=28
    SEED_C=(1 2 3 4 5)
    T_W_PATH=$WRN28_10_C100_PATH
    EXT_MODEL_CFG=",'width':[160,320,640]"
    N_EXP_PER_GPU=3
    W_BITS=4
    ACT_BITS=4
    CV_FC_A=4
    CV_FC_W=4
    MIXUP_C=('@')

elif [ $EXP -eq 511 ]
then
    EXP_G='abs@quality_calibration_wr28-10-cifar100'
    #'cifar10-raw' 'imagine-cifar10-r44-r1000' 'imagine-cifar10-r44-r5000' 'imagine-cifar10-r44-r10000' 'imagine-cifar10-r44-r15000' 'imagine-cifar10-r44-r20000'
    DATASET_C=('cifar100-raw' 'random-cifar100' 'imagine-cifar100-wr28-10-no_dd_kl_r1000' 'imagine-cifar100-wr28-10-dd-exp_kl_r1000_k5s0.375' 'imagine-cifar100-wr28-10-dd-exp_r1000_k5s0.375')
    EXT_FLAGS="--recalibrate --calibration-set-size 5 --shuffle-calibration-steps 200"
    BATCH_SIZE=256
    DEPTH=28
    SEED_C=(1 2 3 4 5)
    T_W_PATH=$WRN28_10_C100_PATH
    EXT_MODEL_CFG=",'width':[160,320,640]"
    N_EXP_PER_GPU=3
    W_BITS=4
    ACT_BITS=4
    CV_FC_A=8
    CV_FC_W=8
    MIXUP_C=('@')
elif [ $EXP -eq 512 ]
then
    EXP_G='abs@quality_calibration_wr28-10-cifar100'
    #'cifar10-raw' 'imagine-cifar10-r44-r1000' 'imagine-cifar10-r44-r5000' 'imagine-cifar10-r44-r10000' 'imagine-cifar10-r44-r15000' 'imagine-cifar10-r44-r20000'
    DATASET_C=('cifar100-raw' 'random-cifar100' 'imagine-cifar100-wr28-10-no_dd_kl_r1000' 'imagine-cifar100-wr28-10-dd-exp_kl_r1000_k5s0.375' 'imagine-cifar100-wr28-10-dd-exp_r1000_k5s0.375')
    EXT_FLAGS="--recalibrate --calibration-set-size 5 --shuffle-calibration-steps 200"
    BATCH_SIZE=256
    DEPTH=28
    SEED_C=(1 2 3 4 5)
    T_W_PATH=$WRN28_10_C100_PATH
    EXT_MODEL_CFG=",'width':[160,320,640]"
    N_EXP_PER_GPU=3
    W_BITS=8
    ACT_BITS=8
    CV_FC_A=8
    CV_FC_W=8
    MIXUP_C=('@')
elif [ $EXP -eq 513 ]
then
    EXP_G='abs@quality_calibration_wr28-10-cifar100'
    #'cifar10-raw' 'imagine-cifar10-r44-r1000' 'imagine-cifar10-r44-r5000' 'imagine-cifar10-r44-r10000' 'imagine-cifar10-r44-r15000' 'imagine-cifar10-r44-r20000'
    DATASET_C=('cifar100-raw' 'random-cifar100' 'imagine-cifar100-wr28-10-no_dd_kl_r1000' 'imagine-cifar100-wr28-10-dd-exp_kl_r1000_k5s0.375' 'imagine-cifar100-wr28-10-dd-exp_r1000_k5s0.375')
    EXT_FLAGS="--recalibrate --calibration-set-size 5 --shuffle-calibration-steps 200"
    BATCH_SIZE=256
    DEPTH=28
    SEED_C=(1 2 3 4 5)
    T_W_PATH=$WRN28_10_C100_PATH
    EXT_MODEL_CFG=",'width':[160,320,640]"
    N_EXP_PER_GPU=3
    W_BITS=4
    ACT_BITS=8
    CV_FC_A=8
    CV_FC_W=8
    MIXUP_C=('@')
elif [ $EXP -eq 514 ]
then
    EXP_G='abs@quality_calibration_wr28-10-cifar100'
    #'cifar10-raw' 'imagine-cifar10-r44-r1000' 'imagine-cifar10-r44-r5000' 'imagine-cifar10-r44-r10000' 'imagine-cifar10-r44-r15000' 'imagine-cifar10-r44-r20000'
    DATASET_C=('cifar100-raw' 'random-cifar100' 'imagine-cifar100-wr28-10-no_dd_kl_r1000' 'imagine-cifar100-wr28-10-dd-exp_kl_r1000_k5s0.375' 'imagine-cifar100-wr28-10-dd-exp_r1000_k5s0.375')
    EXT_FLAGS="--recalibrate --calibration-set-size 5 --shuffle-calibration-steps 200"
    BATCH_SIZE=256
    DEPTH=28
    SEED_C=(1 2 3 4 5)
    T_W_PATH=$WRN28_10_C100_PATH
    EXT_MODEL_CFG=",'width':[160,320,640]"
    N_EXP_PER_GPU=3
    W_BITS=8
    ACT_BITS=4
    CV_FC_A=8
    CV_FC_W=8
    MIXUP_C=('@')
elif [ $EXP -eq 52 ]
then
    EXP_G='abs@quality_calibration_r18-imagenet_new'
    N_EXP_PER_GPU=1
    DATASET_C=('imagenet' 'random-imagenet' 'imagine-imagenet-r18-no_dd_kl_r1000' 'imagine-imagenet-r18_dd-exp_kl_r1000_k5s1' 'imagine-imagenet-r18_dd-exp_r1000_k5s1')
    EXT_FLAGS="--recalibrate --calibration-set-size 10 --shuffle-calibration-steps 200"
    T_W_PATH=$R18_IMGNT_PATH
    DEPTH=18
    BATCH_SIZE=256
    W_BITS=4
    ACT_BITS=4
    CV_FC_A=8
    CV_FC_W=8
    SEED_C=(1 2 3 4 5)
    MIXUP_C=('@')

elif [ $EXP -eq 521 ]
then
    EXP_G='abs@quality_calibration_r18-imagenet_new'
    N_EXP_PER_GPU=1
    DATASET_C=('imagenet' 'random-imagenet' 'imagine-imagenet-r18-no_dd_kl_r1000' 'imagine-imagenet-r18_dd-exp_kl_r1000_k5s1' 'imagine-imagenet-r18_dd-exp_r1000_k5s1')
    #'imagenet' 'random-imagenet' 'imagine-imagenet-r18-no_dd_kl_r1000' 'imagine-imagenet-r18_dd-exp_kl_r1000')
    EXT_FLAGS="--recalibrate --calibration-set-size 10 --shuffle-calibration-steps 200"
    T_W_PATH=$R18_IMGNT_PATH
    DEPTH=18
    BATCH_SIZE=256
    W_BITS=4
    ACT_BITS=8
    CV_FC_A=8
    CV_FC_W=8
    SEED_C=(1 2 3 4 5)
    MIXUP_C=('@')

elif [ $EXP -eq 522 ]
then
    EXP_G='abs@quality_calibration_r18-imagenet_new'
    N_EXP_PER_GPU=1
    DATASET_C=('imagenet' 'random-imagenet' 'imagine-imagenet-r18-no_dd_kl_r1000' 'imagine-imagenet-r18_dd-exp_kl_r1000_k5s1' 'imagine-imagenet-r18_dd-exp_r1000_k5s1')
    EXT_FLAGS="--recalibrate --calibration-set-size 10 --shuffle-calibration-steps 200"
    T_W_PATH=$R18_IMGNT_PATH
    DEPTH=18
    BATCH_SIZE=256
    W_BITS=8
    ACT_BITS=8
    CV_FC_A=8
    CV_FC_W=8
    SEED_C=(1 2 3 4 5)
    MIXUP_C=('@')

elif [ $EXP -eq 523 ]
then
    EXP_G='abs@quality_calibration_r18-imagenet_8w8a'
    N_EXP_PER_GPU=1
    DATASET_C=('imagine-imagenet-r18-dd-exp_r1000')
    #'imagenet' 'random-imagenet' 'imagine-imagenet-r18-no_dd_kl_r1000' 'imagine-imagenet-r18_dd-exp_kl_r1000')
    EXT_FLAGS="--recalibrate --calibration-set-size 10 --shuffle-calibration-steps 200"
    T_W_PATH=$R18_IMGNT_PATH
    DEPTH=18
    BATCH_SIZE=256
    W_BITS=8
    ACT_BITS=8
    CV_FC_A=8
    CV_FC_W=8
    SEED_C=(1 2 3 4 5)
elif [ $EXP -eq 6 ]
then
    EXP_G='abs@cifar10_distilation'
    DATASET_C=('cifar10-raw' 'imagine-cifar10-r44-no_dd_sym_r1000' 'imagine-cifar10-r44-dd-exp_kl_r1000_k5s1' 'imagine-cifar10-r44-dd-exp_r1000_k5s1' )
    DATA_SIZE_C=(50)
elif [ $EXP -eq 0 ]
then
    EXP_G='compare_gen_methods'
    DATASET_C=('imagine-cifar10-r44-no_dd_r100' 'imagine-cifar10-r44-no_dd_r500' 'imagine-cifar10-r44-no_dd_r1000')
    DATA_SIZE_C=(2000) #(4000 5000 6000)
    AUX_C=('smoothl1')
    AUX_SCALE_C=(0.01)
    SEED_C=(1 2 3 4 5)
elif [ $EXP -eq 9 ]
then
    EXP_G='compare_gen_methods'
    DATASET_C=('imagine-cifar10-r44-dd_r100' 'imagine-cifar10-r44-dd_r500' 'imagine-cifar10-r44-dd_r1000')
    DATA_SIZE_C=(1000 2000)
    AUX_C=('smoothl1')
    AUX_SCALE_C=(0.01)
    SEED_C=(1) #2 3 4 5)

elif [ $EXP -eq 123 ]
then
    EXP_G='cifar-100'
    DATASET_C=('imagine-cifar100-wr28-10-no_dd_r500')
    DATA_SIZE_C=(100)
    AUX_C=('smoothl1')
    AUX_SCALE_C=(0.01)
    LR_C=(10 50)
    SEED_C=(1) #2 3 4 5)
    T_W_PATH=results/wresnet28-10_ba_m10/checkpoint.pth.tar
    EXT_MODEL_CFG=",'width':[160,320,640]"
    DEPTH=28
    BATCH_SIZE=256
    W_BITS=4
    W_RANGE_MOD=
elif [ $EXP -eq 666 ]
then
    #cross over cifar10 datasets to cifat100
    EXP_G='cross-over_cifar10-cifar-100'
    DATASET_C=('crossover@cifar100@cifar10-raw' 'crossover@cifar100@imagine-cifar10-r44-no_dd_r1000' )
    DATA_SIZE_C=(5000)
    AUX_C=('smoothl1')
    AUX_SCALE_C=(0.01)
    LR_C=(50)
    SEED_C=(1) #2 3 4 5)
    T_W_PATH=results/wresnet28-10_ba_m10/checkpoint.pth.tar
    EXT_MODEL_CFG=",'width':[160,320,640]"
    DEPTH=28
    BATCH_SIZE=256
    W_BITS=4
    W_RANGE_MOD=
elif [ $EXP -eq 999 ]
then
    #cross over cifar10 datasets to stl,svhn
    N_EXP_PER_GPU=2
    EXP_G='abs@cross-over-cifar10'
    DATASET_C=('crossover@cifar10@stl10' 'crossover@cifar10@SVHN') #crossover@cifar100@cifar10-raw'  'crossover@cifar100@imagine-cifar10-r44-no_dd_r1000' )
    DATA_SIZE_C=('@')
    AUX_C=('smoothl1')
    AUX_SCALE_C=(0.01)
    LR_C=(100)
    SEED_C=(1 2 3)
    T_W_PATH=$R44_C10_PATH
    DEPTH=44
    BATCH_SIZE=512
    W_BITS=2
else
    echo '> experiment ID not defined'
    exit 0
fi
let EXP_HOLD_EVERY=$N_EXP_PER_GPU*${#DEVICE_ID_C[@]}
echo "> running up to $EXP_HOLD_EVERY experiments in parallel"
let TOT_EXP_TO_RUN=`echo "${#MIXUP_C[@]}*${#BN_MOD_C[@]}*${#DATA_SIZE_C[@]}*${#AUX_C[@]}*${#AUX_SCALE_C[@]}*${#LR_C[@]}*${#LOSS_C[@]}*${#SEED_C[@]}"`
echo '>' running exp set $EXP with $TOT_EXP_TO_RUN configurations
EXP_COUNT=0
for SEED in ${SEED_C[@]}
do
for DATASET in ${DATASET_C[@]}
do
    for SIZE in ${DATA_SIZE_C[@]}
    do
        for BN_MOD in ${BN_MOD_C[@]}
        do
        for LOSS in ${LOSS_C[@]}
        do
        for AUX in ${AUX_C[@]}
        do
        for MIX in ${MIXUP_C[@]}
        do
        for LR in ${LR_C[@]}
        do
        for AUX_S in ${AUX_SCALE_C[@]}
        do
            echo init allocation loop
            # find a free gpu for experiment
            GPU_ID=0
            GPU=${DEVICE_ID_C[0]}
            esc=1
            while [ $esc ]
            do
                echo '>' enter gpu allocation block
                tmp=$RUNNING_PROCS
                for i in ${RUNNING_PROCS[@]:1}; do tmp="$tmp|$i"; done
                echo $tmp
                #check how many processes where lunched
                while [ ${#RUNNING_PROCS[@]} -gt 0 -a `nvidia-smi | grep -E "$tmp" | wc -l` -ge $EXP_HOLD_EVERY ]
                do
                    #t=`bc -l <<< "scale=4; 5+${RANDOM}/32767"`
                    #sleep $t
                    sleep 5
                done

                while [ `nvidia-smi | grep python | grep " $GPU " | wc -l` -ge $N_EXP_PER_GPU ]
                do
                    let GPU_ID=$GPU_ID+1
                    let GPU_ID=$GPU_ID%${#DEVICE_ID_C[@]}
                    GPU=${DEVICE_ID_C[$GPU_ID]}
                    echo '>' gpu "$GPU" is running `nvidia-smi | grep python| grep " $GPU "| wc -l` python tasks
                    if [ $GPU_ID -eq 0 ]
                    then sleep 5
                    fi
                done
                echo '>' found open GPU "$GPU"

                if [ `nvidia-smi | grep python| grep " $GPU " | wc -l` -lt $N_EXP_PER_GPU ]
                then
                    echo '>' leaving gpu allocation block
                    esc=0
                    break
                fi
            done

            # get valid experiment flags
            AUX_=`getarg $AUX "--aux"`
            AUX_S_=`getarg $AUX_S "--aux-loss-scale"`
            MIX_=`getarg $MIX`
            SIZE_=`getarg $SIZE "--dist-set-size"`
            LR_=`getarg $LR ",'scale_lr':"`
            BN_MOD_=`getarg $BN_MOD`

            echo $EXP_G
            echo dataset: $DATASET
            echo $SEED
            echo samples per class: $SIZE
            echo bn mod: $BN_MOD
            echo loss: $LOSS
            echo aux: $AUX
            echo mixup: $MIX
            echo lr_scale: $LR
            echo aux_scale: $AUX_S

            M_CFG="{'depth': $DEPTH, 'regime':'sgd_cos_staggerd_1', 'conv1':{'a': $CV_FC_A,'w':$CV_FC_W},'fc':{'a': $CV_FC_A,'w':$CV_FC_W}, 'activations_numbits':$ACT_BITS,'weights_numbits':$W_BITS, 'bias_quant':False $LR_ $EXT_MODEL_CFG}"
            echo '>' start experiment $EXP_COUNT' / '$TOT_EXP_TO_RUN, cfg: $M_CFG
            eval python qdistiler_main.py --model resnet --model_config '"$M_CFG"'  --teacher $T_W_PATH --dataset $DATASET \
            --train-first-conv --steps-per-epoch $STEPS_PER_EPOCH --quant-freeze-steps -1 -b $BATCH_SIZE --device_ids $GPU_ID $W_RANGE_MOD \
            $SIZE_ --exp-group $EXP_G $AUX_ --kd-loss $LOSS $STEP_LIMIT $BN_MOD_ $AUX_S_ $MIX_\
            $EXT_FLAGS --print-freq 100 --seed $SEED 2> $EXP_RUNNER_LOGDIR/$EXP'_'$EXP_COUNT'_stderr.log' &
            EXP_PROC="$!"
            echo "> pid for exp $EXP_COUNT $EXP_PROC"
            let EXP_COUNT=$EXP_COUNT+1
            RUNNING_PROCS+=("$EXP_PROC")
            #sleep to avoid result dir overwrite
            sleep 10

            let GPU_ID=$GPU_ID+1
            let GPU_ID=$GPU_ID%${#DEVICE_ID_C[@]}
            GPU=${DEVICE_ID_C[$GPU_ID]}

        done #AUX_S
        done #LR
        done #MIX
        done #AUX
        done #LOSS
        done #BN
    done #SIZE
    done #SEED
done #DATASET

#wait untill done
echo 'DONE Launching experiments! staying alive for processes to finish running'
tmp=$RUNNING_PROCS
for i in ${RUNNING_PROCS[@]:1}; do tmp="$tmp|$i"; done
while [ `nvidia-smi | grep -E "$tmp" | wc -l` -gt 0 ]
do
   sleep 10
done
set +x
#set +e
