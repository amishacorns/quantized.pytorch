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
echo "> running "$EXP_HOLD_EVERY" experiments in parallel"
T_W_PATH=results/resnet44_cifar10/model_best.pth.tar
#STEP_LIMIT="--steps-limit 16000"
##### default settings
#script accepts experiment ÍD argument
EXP=${1:-1}
DATASET_C=('cifar10-raw')
DATA_SIZE_C=('@')
AUX_C=('@')
BN_MOD_C=('@')
LOSS_C=('mse')
LR_C=(1.)
MIXUP_C=('@')
AUX_SCALE_C=('@')
if [ $EXP -eq 1 ]
then
    ##### EXP=1 - find best lr per loss function
    EXP_G='loss_and_lr'
    LOSS_C=('mse' )
    #LR_C=(50 5 2) loss diverged under these values
    # best 90.9 0.6951928
    LR_C=(0.6951928)
    #####
elif [ $EXP -eq 11 ]
then
    ##### EXP=1 - find best lr per loss function
    EXP_G='loss_and_lr'
    LOSS_C=('smoothl1')
    #LR_C=(50 5 ) loss diverged under these values
    # best 90.13 0.6951928
    #####
elif [ $EXP -eq 12 ]
then
    ##### EXP=1 - find best lr per loss function
    EXP_G='loss_and_lr'
    LOSS_C=('kld')
    LR_C=(100 70 30) #best 91.24 50
    #####

elif [ $EXP -eq 2 ]
then
    ##### EXP=2 -compare results of different batch normalization schemes
    EXP_G='loss_and_lr'
    BN_MOD_C=('--fresh-bn') #,'--absorb-bn')
    LOSS_C=('mse')
    LR_C=(0.6951928)
    #####
elif [ $EXP -eq 21 ]
then
    ##### EXP=2 -compare results of different batch normalization schemes
    EXP_G='loss_and_lr'
    BN_MOD_C=('--fresh-bn') #,'--absorb-bn')
    LOSS_C=('smoothl1')
    LR_C=(1.)
    #####
elif [ $EXP -eq 22 ]
then
    ##### EXP=2 -compare results of different batch normalization schemes
    EXP_G='loss_and_lr'
    BN_MOD_C=('--fresh-bn') #,'--absorb-bn')
    LOSS_C=('kld')
    LR_C=(100)
    #####
elif [ $EXP -eq 3 ]
then
    ##### EXP=3 - compare aux losses + mixup
    EXP_G='aux_and_mixup'
    BN_MOD_C=('@','--fresh-bn') #todo use best from exp 2
    AUX_C=('mse' 'smoothl1' 'cos') #todo search over loss/aux scales?
    LOSS_C=('mse')
    MIXUP_C=('@' '--mixup') #todo test on main loss only
    LR_C=(0.7)
    #####
elif [ $EXP -eq 301 ]
then
    ##### EXP=3 - compare aux losses + mixup
    EXP_G='aux_and_mixup'
    BN_MOD_C=('@','--fresh-bn') #todo use best from exp 2
    AUX_C=('kld') #todo search over loss/aux scales?
    LOSS_C=('mse')
    MIXUP_C=('@' '--mixup') #todo test on main loss only
    LR_C=(0.7)
    #####
elif [ $EXP -eq 31 ]
then
    ##### EXP=3 - compare aux losses + mixup
    EXP_G='aux_and_mixup'
    BN_MOD_C=('@') #todo use best from exp 2
    AUX_C=('kld' ) #todo search over loss/aux scales?
    AUX_SCALE_C=(0.6 0.3 0.06 0.01 0.005)
    LOSS_C=('kld')
    MIXUP_C=('@' '--mixup')
    LR_C=(100)
    #####
elif [ $EXP -eq 311 ]
then
    ##### EXP=3 - compare aux losses + mixup
    EXP_G='aux_and_mixup'
    AUX_C=('mse' 'smoothl1' 'cos')
    LOSS_C=('kld')
    AUX_SCALE_C=(0.01 0.005) #mse tried 0.1, 0.05
    MIXUP_C=('@' '--mixup')
    LR_C=(100)
    #####

elif [ $EXP -eq 4 ]
then
    ##### EXP=4 - compare data impact mixup
    EXP_G='data_size'
    BN_MOD_C=('@') #todo best from exp 2 --fresh-bn
    DATA_SIZE_C=(1 10 50 100 500 1000 2000)
    AUX_C=('smoothl1' 'cos')
    AUX_SCALE_C=(0.01 0.005) #todo best from exp3
    LOSS_C=('kld') #todo best from exp3
    LR_C=(100)
    MIXUP_C=('@' '--mixup') #mixup is generally better
    #####
elif [ $EXP -eq 41 ]
then
    ##### EXP=4 - compare data impact mixup
    EXP_G='data_size'
    BN_MOD_C=('@')
    DATA_SIZE_C=(1 10 50 100 500 1000 2000)
    AUX_C=('kld')
    AUX_SCALE_C=(0.01 0.005)
    LOSS_C=('kld')
    LR_C=(100)
    MIXUP_C=('@' '--mixup')
    #####
else
    echo '> experiment ID not defined'
    exit 0
fi

let TOT_EXP_TO_RUN=${#MIXUP_C[@]}*${#BN_MOD_C[@]}*${#DATA_SIZE_C[@]}*${#AUX_C[@]}*${#AUX_SCALE_C[@]}*${#LR_C[@]}*${#LOSS_C[@]}
echo '>' running exp set $EXP with $TOT_EXP_TO_RUN configurations
EXP_COUNT=0

for DATASET in ${DATASET_C[@]}
do
echo dataset: $DATASET
    for SIZE in ${DATA_SIZE_C[@]}
    do
        echo samples per class: $SIZE

        for BN_MOD in ${BN_MOD_C[@]}
        do
        echo bn mod: $BN_MOD

        for LOSS in ${LOSS_C[@]}
        do
        echo loss: $LOSS

        for AUX in ${AUX_C[@]}
        do
        echo aux: $AUX

        for MIX in ${MIXUP_C[@]}
        do
        echo mixup: $MIX

        for LR in ${LR_C[@]}
        do
        echo lr_scale: $LR

        for AUX_S in ${AUX_SCALE_C[@]}
        do
        echo aux_scale: $AUX_S
        #    if (( ($EXP_COUNT % $EXP_HOLD_EVERY) == 0 ))
        #    then BG='';
        #    else BG='&';
        #    fi;
            BG='&'
            # get valid experiment flags
            AUX_=`getarg $AUX "--aux"`
            AUX_S_=`getarg $AUX_S "--aux-loss-scale"`
            MIX_=`getarg $MIX`
            SIZE_=`getarg $SIZE "--dist-set-size"`
            LR_=`getarg $LR ",'scale_lr':"`
            BN_MOD_=`getarg $BN_MOD`
            echo init allocation loop

            # find a free gpu for experiment
            let GPU_ID=0 # $EXP_COUNT%${#DEVICE_ID_C[@]}
            esc=1
            while [ $esc ]
            do
                echo '>' enter gpu allocation block
                while [ `nvidia-smi | grep python| wc -l` -ge $EXP_HOLD_EVERY ]
                do
                    sleep 5
                done

                while [ `nvidia-smi | grep python | grep " $GPU_ID " | wc -l` -ge $N_EXP_PER_GPU ]
                do
                    let GPU_ID=$GPU_ID+1
                    let GPU_ID=$GPU_ID%${#DEVICE_ID_C[@]}
                    sleep 1
                    echo '>' gpu "$GPU_ID" is running `nvidia-smi | grep python| grep " $GPU_ID "| wc -l` python tasks
                done
                echo '>' found open GPU "$GPU_ID"

                if [ `nvidia-smi | grep python| grep " $GPU_ID " | wc -l` -lt $N_EXP_PER_GPU ]
                then
                    echo '>' leaving gpu allocation block
                    esc=0
                    break
                fi
            done

            M_CFG="{'depth': 44, 'regime':'sgd_cos_staggerd_1', 'conv1':{'a': 4,'w':4},'fc':{'a': 4,'w':4}, 'activations_numbits':4,'weights_numbits':2, 'bias_quant':False $LR_}"
            echo '>' start experiment $EXP_COUNT' / '$TOT_EXP_TO_RUN, cfg: $M_CFG
            eval python qdistiler_main.py --model resnet --model_config '"$M_CFG"'  --teacher $T_W_PATH --dataset $DATASET \
            --train-first-conv --steps-per-epoch 200 --quant-freeze-steps -1 --b 512 --device_ids $GPU_ID --free-w-range \
            $SIZE_ --exp-group $EXP_G $AUX_ --loss $LOSS $STEP_LIMIT $BN_MOD_ $AUX_S_ $MIX_\
            --print-freq 100 2> $EXP'_'$EXP_COUNT'_stderr.log' $BG
            echo "> pid for exp $EXP_COUNT $!"
            let EXP_COUNT=$EXP_COUNT+1

            #sleep to avoid result dir overwrite
            sleep 10

        done #AUX_S
        done #LR
        done #MIX
        done #AUX
        done #LOSS
        done #BN
    done #SIZE
done #DATASET

#wait untill done
echo 'DONE Launching experiments! staying alive for processes to finish running'
while [ `nvidia-smi | grep python| wc -l` -ge 0 ]
do
   sleep 10
done
set +x
#set +e