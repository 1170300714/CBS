#!/bin/bash
SEED=1
DATASET=miniImageNet
CFG=vit_b16_kmeans_random_discard_greedy
TRAINER=CoOp_miniImageNet_wo_Base_Our_ACIL_Distribution
R=$1
CLASS_PER_TASK=20
for ((i = 0 ; i < 5 ; i++))
do
    if [ $i -eq 0 ]; then
        python train.py \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir output_acil/${TRAINER}/${DATASET}/${CFG}/R_${R}/session0 \
            TRAINER.TASK_ID 0 \
            TRAINER.CLASS_PER_TASK ${CLASS_PER_TASK} \
            AL.SAMPLE_SAVE_PATH miniimagenet/acil/distribution_kmeans_random_discard_greedy/R_${R}/SEED1 \
            AL.ROUND ${R} 
    else
        j=$(($i-1))
        python train.py \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir output_acil/${TRAINER}/${DATASET}/${CFG}/R_${R}/session${i} \
            --model-dir output_acil/${TRAINER}/${DATASET}/${CFG}/R_${R}/session${j} \
            TRAINER.TASK_ID ${i} \
            TRAINER.CLASS_PER_TASK ${CLASS_PER_TASK} \
            AL.SAMPLE_SAVE_PATH miniimagenet/acil/distribution_kmeans_random_discard_greedy/R_${R}/SEED1 \
            AL.ROUND ${R} 
    fi
done