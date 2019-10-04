#!/usr/bin/env bash

python test_models.py hmdb51 Flow models/hmdb51_fc_4conv_rgb/hmdb51_fc_4conv_rgb_model_best.pth.tar --gpus 0 -j 0
