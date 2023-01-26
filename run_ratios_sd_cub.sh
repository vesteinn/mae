for rat in 0 0.1 0.3 0.5 0.7 0.8 0.9 1.0; do

JOB_DIR="augmentation_ratio_large_${rat}_cub_gen_finetune_sd_only_filterfix"
PRETRAIN_CHKPT="mae_pretrain_vit_large.pth"
DATA_DIR="../cub_w_sd"

python submitit_finetune.py \
    --job_dir ${JOB_DIR} \
    --ngpus 4 \
    --nodes 1 \
    --accum_iter 3 \
    --batch_size 12 \
    --model vit_large_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --timeout 300 \
    --cls_token \
    --epochs 50 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${DATA_DIR} \
    --keep_orig_ratio $rat

done


