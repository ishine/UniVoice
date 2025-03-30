#!/bin/bash 
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 36501 univoice/infer/infer_asr.py \
    --ckpt_path univoice_all \
    --temperature 0.7 \
    --top_p 0.95 \
    --top_k 50 \
    --num_beams 4

