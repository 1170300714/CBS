#!/bin/bash
SEED=2
DATASET=cub200
CFG=vit_b16
TRAINER=CoOp_CUB_wo_Base_Our_ACIL_Random
R=5
CLASS_PER_TASK=20
for ((i = 0 ; i < 10 ; i++))
do
    if [ $i -eq 0 ]; then
        python train.py \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir output_acil/${TRAINER}/${DATASET}/${CFG}/R_${R}/SEED${SEED}/session0 \
            TRAINER.TASK_ID 0 \
            TRAINER.CLASS_PER_TASK ${CLASS_PER_TASK} \
            AL.SAMPLE_SAVE_PATH CUB_200_2011/ACIL_margin/R_${R}/SEED1 \
            AL.ROUND ${R}
    else
        j=$(($i-1))
        python train.py \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir output_acil/${TRAINER}/${DATASET}/${CFG}/R_${R}/SEED${SEED}/session${i} \
            --model-dir output_acil/${TRAINER}/${DATASET}/${CFG}/R_${R}/SEED${SEED}/session${j} \
            TRAINER.TASK_ID ${i} \
            TRAINER.CLASS_PER_TASK ${CLASS_PER_TASK} \
            AL.SAMPLE_SAVE_PATH CUB_200_2011/ACIL_margin/R_${R}/SEED1 \
            AL.ROUND ${R}
    fi
done