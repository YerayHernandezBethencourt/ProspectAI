#!/bin/bash

MODEL="openbmb/MiniCPM-Llama3-V-2_5" 
DATA="train_anotations.json" # json file
EVAL_DATA="val_anotations.json" # json file
LLM_TYPE="llama3" 

python -m transformers.trainer \
    run_clm.py \
    --model_name_or_path "$MODEL" \
    --train_file "$DATA" \
    --validation_file "$EVAL_DATA" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --output_dir "./results" \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \
    --save_steps 10000 \
    --logging_steps 1000 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --deepspeed ds_config.json