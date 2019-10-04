#!/bin/bash
LOG="logs/hmdb51_fc_4conv_rgb$RANK.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python -u main.py hmdb51 RGB \
	splits/hmdb51/hmdb51_train_split_1_total.txt \
        splits/hmdb51/hmdb51_val_split_1_total.txt \
	--b 64 --gd 20 -j 0 --dropout 0.5 \
	--lr 0.01 --lr_steps 2 4 6 8 --epochs 8 \
	--snapshot_pref hmdb51_fc_4conv_rgb \
	--finetune pretrain/kinetics_rgb.pth \
    --gpus 0
