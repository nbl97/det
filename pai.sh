#!/bin/bash
export MASTER_ADDR=$PAI_HOST_IP_taskrole_0
export MASTER_PORT=23456
export NODE_RANK=$PAI_TASK_INDEX
echo $MASTER_ADDR
echo $MASTER_PORT

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT tools/train.py configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py \
  --work-dir /mnt/configblob/users/bolin/s2b_22302_det/
