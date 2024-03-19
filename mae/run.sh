# JOB_DIR="/home/data02/zjc/abaw/mae/output"
# # 
# IMAGENET_DIR="/home/data02/zjc/abaw/dataset/Unity"


# torchrun main_finetune.py \
#     --job_dir ${JOB_DIR} \
#     --nodes 1 \
#     --batch_size 4 \
#     --model vit_base_patch16 \
#     --finetune ${PRETRAIN_CHKPT} \
#     --epochs 100 \
#     --blr 5e-4 --layer_decay 0.65 \
#     --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#     --dist_eval --data_path ${IMAGENET_DIR}




# # base
# IMAGENET_DIR="/home/data02/zjc/abaw/dataset/Unity"
# # PRETRAIN_CHKPT="/home/data02/zjc/abaw/mae/pretrained/mae_pretrain_vit_base.pth"
# PRETRAIN_CHKPT='/home/data02/zjc/abaw/mae/output_dir_base_unity/checkpoint-84.pth'
# torchrun --nproc_per_node 1 main_finetune.py \
#     --batch_size 128 \
#     --model vit_base_patch16 \
#     --nb_classes 8\
#     --finetune ${PRETRAIN_CHKPT} \
#     --epochs 100 \
#     --blr 5e-4 --layer_decay 0.65 \
#     --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#     --dist_eval --data_path ${IMAGENET_DIR}\
#     --eval







# # raddb -ce finetune 78.09
# JOB_DIR="/home/data02/zjc/abaw/mae/output"
# IMAGENET_DIR="/home/data02/zjc/abaw/dataset/RAFDBCE7"
# PRETRAIN_CHKPT="/home/data02/zjc/abaw/mae/output_dir_rafdbce7/checkpoint-46.pth"
# CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node 1 finetune_rafdbce.py \
#     --batch_size 64 \
#     --nb_classes 7\
#     --model vit_base_patch16  \
#     --finetune ${PRETRAIN_CHKPT} \
#     --epochs 100 \
#     --blr 1e-3 --layer_decay 0.75 \
#     --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#     --dist_eval --data_path ${IMAGENET_DIR}\
#     --eval






# raddb -ce finetune 78.09
JOB_DIR="/home/data02/zjc/abaw/mae/output"
IMAGENET_DIR="/home/data02/zjc/abaw/dataset/RAFDBCE7"
PRETRAIN_CHKPT="/home/data02/zjc/abaw/mae/output_dir_rafdbce7/checkpoint-46.pth"
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node 1 finetune_rafdbce.py \
    --batch_size 64 \
    --nb_classes 7\
    --model vit_base_patch16  \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR}\
    --eval --pred
