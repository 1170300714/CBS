SEED=$1
TRAINER=CoOp_CUB_wo_Base_Our_ACIL_Random
OUTPUT_DIR=CoOp_CUB_wo_Base_ACIL_Our_random_force_balance_10_session_R_5_seed${SEED}
scripts=main_cub_wo_base_our_acil_random
CFG=vit_b16_force_balance
SAMPLE_SAVE_PATH=CUB_200_2011/ACIL_random_force_balance/R_5/SEED${SEED}
bash scripts/acil_coop/${scripts}.sh cub200 ${CFG} end 16 5 False 0 None 100 100 20 ${SEED} ${SAMPLE_SAVE_PATH}
bash scripts/acil_coop/${scripts}.sh cub200 ${CFG} end 16 5 False 1 output_acil/${OUTPUT_DIR}/${TRAINER}/${CFG}/nctx16_cscFalse_ctpend/task0/seed${SEED}  100 100 20 ${SEED} ${SAMPLE_SAVE_PATH}
bash scripts/acil_coop/${scripts}.sh cub200 ${CFG} end 16 5 False 2 output_acil/${OUTPUT_DIR}/${TRAINER}/${CFG}/nctx16_cscFalse_ctpend/task1/seed${SEED}  100 100 20 ${SEED} ${SAMPLE_SAVE_PATH}
bash scripts/acil_coop/${scripts}.sh cub200 ${CFG} end 16 5 False 3 output_acil/${OUTPUT_DIR}/${TRAINER}/${CFG}/nctx16_cscFalse_ctpend/task2/seed${SEED}  100 100 20 ${SEED} ${SAMPLE_SAVE_PATH}
bash scripts/acil_coop/${scripts}.sh cub200 ${CFG} end 16 5 False 4 output_acil/${OUTPUT_DIR}/${TRAINER}/${CFG}/nctx16_cscFalse_ctpend/task3/seed${SEED}  100 100 20 ${SEED} ${SAMPLE_SAVE_PATH}
bash scripts/acil_coop/${scripts}.sh cub200 ${CFG} end 16 5 False 5 output_acil/${OUTPUT_DIR}/${TRAINER}/${CFG}/nctx16_cscFalse_ctpend/task4/seed${SEED}  100 100 20 ${SEED} ${SAMPLE_SAVE_PATH}
bash scripts/acil_coop/${scripts}.sh cub200 ${CFG} end 16 5 False 6 output_acil/${OUTPUT_DIR}/${TRAINER}/${CFG}/nctx16_cscFalse_ctpend/task5/seed${SEED}  100 100 20 ${SEED} ${SAMPLE_SAVE_PATH}
bash scripts/acil_coop/${scripts}.sh cub200 ${CFG} end 16 5 False 7 output_acil/${OUTPUT_DIR}/${TRAINER}/${CFG}/nctx16_cscFalse_ctpend/task6/seed${SEED}  100 100 20 ${SEED} ${SAMPLE_SAVE_PATH}
bash scripts/acil_coop/${scripts}.sh cub200 ${CFG} end 16 5 False 8 output_acil/${OUTPUT_DIR}/${TRAINER}/${CFG}/nctx16_cscFalse_ctpend/task7/seed${SEED}  100 100 20 ${SEED} ${SAMPLE_SAVE_PATH}
bash scripts/acil_coop/${scripts}.sh cub200 ${CFG} end 16 5 False 9 output_acil/${OUTPUT_DIR}/${TRAINER}/${CFG}/nctx16_cscFalse_ctpend/task8/seed${SEED}  100 100 20 ${SEED} ${SAMPLE_SAVE_PATH}
# bash scripts/coop/${scripts}.sh cub200 vit_b16 end 16 5 False 9 None 100 100 20

