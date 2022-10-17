#!/bin/bash

export PYTHONPATH=./
# initialize the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pt171  # pytorch env
PYTHON=python

# source /etc/proxyrc # To have internet access

# DATA=/local/${SLURM_JOB_ID}/ilsvrc2012
DATA=/scratch/1/data/ilsvrc2012/untarred-version/

# echo Copying imagenet
# mkdir $DATA
# scp -r  /scratch/1/data/ilsvrc2012/untarred-version/* $DATA

# echo Finished copying imagenet

#--------------

EPSILON=$1
ALPHA=$2
UNIF=$3
CLIP=$4

CONFIG1=configs/configs_fast_phase1_short_all.yml
CONFIG2=configs/configs_fast_phase2_all.yml
CONFIG3=configs/configs_fast_phase3_all.yml
CONFIG_EVAL=configs/configs_fast_evaluate_all.yml

PREFIX1=fast_adv_phase1_
PREFIX2=fast_adv_phase2_
PREFIX3=fast_adv_phase3_

END1=/scratch/1/project/co_neurips/debug/fast_adv_models/${PREFIX1}epsilon${EPSILON}_alpha${ALPHA}_unif${UNIF}_clip${CLIP}/checkpoint_epoch1.pth.tar
END2=/scratch/1/project/co_neurips/debug/fast_adv_models/${PREFIX2}epsilon${EPSILON}_alpha${ALPHA}_unif${UNIF}_clip${CLIP}/checkpoint_epoch12.pth.tar
END3=/scratch/1/project/co_neurips/debug/fast_adv_models/${PREFIX3}epsilon${EPSILON}_alpha${ALPHA}_unif${UNIF}_clip${CLIP}/checkpoint_epoch15.pth.tar

# training for phase 1
python -u main_fast.py $DATA -c $CONFIG1 --epsilon $EPSILON --alpha $ALPHA --unif $UNIF --clip $CLIP --output_prefix $PREFIX1

# training for phase 2
python -u main_fast.py $DATA -c $CONFIG2 --resume $END1 --epsilon $EPSILON --alpha $ALPHA --unif $UNIF --clip $CLIP --output_prefix $PREFIX2

# training for phase 3
python -u main_fast.py $DATA -c $CONFIG3 --resume $END2 --epsilon $EPSILON --alpha $ALPHA --unif $UNIF --clip $CLIP --output_prefix $PREFIX3

# evaluation
python -u main_fast.py $DATA -c $CONFIG_EVAL --resume $END3 --evaluate --restarts 10 --epsilon $EPSILON --alpha $ALPHA --unif $UNIF --clip $CLIP --output_prefix $PREFIX3
