#!/bin/bash
SEED=1
DATASET=dtd
CFG=vit_b16_typiclust
TRAINER=CoOp_DTD_wo_Base_Our_ACIL_Exist_AL
R=$1
CLASS_PER_TASK=20
for ((i = 0 ; i < 2 ; i++))
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
            AL.SAMPLE_SAVE_PATH dtd/acil/Typiclust/R_${R}/SEED1 \
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
            AL.SAMPLE_SAVE_PATH dtd/acil/Typiclust/R_${R}/SEED1 \
            AL.ROUND ${R}
    fi
done