#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate p37t1
#@ is used to represent a None like value that ignores the parameter (ie not used). return second argument as prefix
function getarg() {
if [ $1 == '@' ];
then echo '';
else echo $2 $1;
fi; }

#set -x
#set -e
#general settings
DEVICE_ID_C=(0 1 2 3)
N_EXP_PER_GPU=2
#assuming all experiments take the same time
let EXP_HOLD_EVERY=$N_EXP_PER_GPU*${#DEVICE_ID_C[@]}
echo "running :"$EXP_HOLD_EVERY" experiments in parallel"
T_W_PATH=results/resnet44_cifar10/model_best.pth.tar
#STEP_LIMIT="--steps-limit 16000"
##### default settings
#script accepts experiment ÍD argument
EXP=${1:-1}

DATA_SIZE_C=('@')
AUX_C=('@')
BN_MOD_C=('@')
LOSS_C=('mse')
LR_C=(1.)
MIXUP_C=('@')
if [ $EXP -eq 1 ]
then
    ##### EXP=1 - find best lr per loss function
    EXP_G='loss_and_lr'
    LOSS_C=('mse' 'smoothl1')
    LR_C=(10 5 2 1.5 1.3 1. 0.6951928 0.55 0.48329302 0.33598183 )
    #####
elif [ $EXP -eq 11 ]
then
    ##### EXP=1 - find best lr per loss function
    EXP_G='loss_and_lr'
    LOSS_C=('kld')
    LR_C=(100 80 60 50 30 20)
    #####
elif [ $EXP -eq 2 ]
then
    ##### EXP=2 -compare results of different batch normalization schemes
    EXP_G='bn_mode_and_loss'
    BN_MOD_C=('@','--fresh-bn','--absorb-bn')
    LOSS_C=('mse' 'smoothl1' 'kld')
    LR_C=(.5) #todo maybe fix best lr to loss
    #####
elif [ $EXP -eq 3 ]
then
    ##### EXP=3 - compare aux losses + mixup
    EXP_G='aux_and_mixup'
    BN_MOD_C=('@') #todo use best from exp 2
    AUX_C=('mse' 'smoothl1' 'kld' 'cos') #todo search over loss/aux scales?
    LOSS_C=('mse' 'smoothl1' 'kld') #todo choose only 2
    MIXUP_C=('@' '--mixup') #todo test on main loss only
    LR_C=( 0.59948425 0.35938137 0.21544347 0.12915497
           0.07742637 0.04641589 0.02782559 0.01668101 0.01) #todo choose 3
    #####
elif [ $EXP -eq 4 ]
then
    ##### EXP=4 - compare data impact mixup
    EXP_G='data_size'
    DATA_SIZE_C='1 10 50 100 500 1000 2000'
    BN_MOD_C=('@') #todo use best from exp 2
    AUX_C=('mse' 'smoothl1' 'kld' 'cos') #todo best from exp3
    LOSS_C=('mse' 'smoothl1' 'kld') #todo best from exp3
    MIXUP_C=('@' '--mixup')
    #####
else
    echo 'experiment ID not defined'
    exit 0
fi


EXP_COUNT=0
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

    let GPU_ID=($EXP_COUNT % ${#DEVICE_ID_C[@]})

    AUX_=`getarg $AUX "--aux"`
    MIX_=`getarg $MIX`
    SIZE_=`getarg $SIZE "--dist-set-size"`
    LR_=`getarg $LR ",'scale_lr':"`
    BN_MOD_=`getarg $BN_MOD`

#    let EXP_COUNT=($EXP_COUNT+1)
#    if (( ($EXP_COUNT % $EXP_HOLD_EVERY) == 0 ))
#    then BG='';
#    else BG='&';
#    fi;
    BG='&'
    M_CFG="{'depth': 44, 'regime':'sgd_cos_staggerd_1', 'conv1':{'a': 4,'w':4},'fc':{'a': 4,'w':4}, 'activations_numbits':4,'weights_numbits':2, 'bias_quant':False $LR_}"
    echo start experiment: size-$SIZE mix-$MIX loss-$LOSS aux-$AUX gpu_id-$GPU_ID scale-$LR bn-$BN
    #eval echo '"$M_CFG"'
    eval python qdistiler_main.py --model resnet --model_config '"$M_CFG"'  --teacher $T_W_PATH --dataset cifar10-raw --train-first-conv \
    --steps-per-epoch 200 --quant-freeze-steps -1 --b 512 --device_ids $GPU_ID $MIX_ --free-w-range \
    $SIZE_ --exp-group $EXP_G $AUX_ --loss $LOSS $STEP_LIMIT --print-freq 100 2> $EXP_COUNT"_stderr.log" $BG

    let EXP_COUNT=$EXP_COUNT+1
    #sleep to avoid result dir overwrite
    sleep 5
    # wait untill resources are available
    while [ `nvidia-smi | grep python| wc -l` -ge $EXP_HOLD_EVERY ]
    do
       sleep 10
    done

    done #LR
    done #MIX
    done #AUX
    done #LOSS
    done #BN
done #SIZE
#set +x
#set +e