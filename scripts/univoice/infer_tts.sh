#!/bin/bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 36505 --nproc_per_node 1 univoice/infer/infer_tts.py \
    --ckpt_path univoice_all \
    --cfg_scale 3
