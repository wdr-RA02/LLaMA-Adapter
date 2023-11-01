#!/bin/bash

LLAMA_PATH="/home/mas-xie.haojie/codes_gzp/datas/LLaMA-7B"
CKPT_PATH="$1"
OUTPUT_DIR="$2"
TYPE="${3:-LORA}"

python convert.py --llama_dir $LLAMA_PATH \
                  --checkpoint $CKPT_PATH \
                  --output_dir $OUTPUT_DIR \
                  --type $TYPE
                  