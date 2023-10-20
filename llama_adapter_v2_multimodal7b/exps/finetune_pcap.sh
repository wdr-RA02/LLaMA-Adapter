#!/bin/bash

run_time=$(date +%Y%m%d-%H%M%S)

misc_root="/home/mas-xie.haojie/codes_gzp/datas/llama_adapter_misc"
LLAMA_PATH="/home/mas-xie.haojie/codes_gzp/datas/LLaMA-7B"
PRETRAINED_PATH="$misc_root/LLaMA-Adapter_LORA-BIAS-7B-v21.pth" # path to pre-trained checkpoint
CONFIG="data/pcap.yaml"
OUTPUT_DIR="$misc_root/output/$run_time"

mkdir -p "$OUTPUT_DIR"

torchrun --nproc-per-node gpu \
 main_finetune.py --data_config "$CONFIG" --batch_size 2 \
 --epochs 4 --warmup_epochs 1 --blr 10e-4 --weight_decay 0.02 \
 --num_workers 2 \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR" \
 --pretrained_path "$PRETRAINED_PATH" 