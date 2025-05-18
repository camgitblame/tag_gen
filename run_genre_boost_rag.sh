#!/bin/bash
# === TRAIN ===
PYTHONPATH=.. python scripts/run_clm_genre.py \
  --model_name_or_path gpt2 \
  --tokenizer_name tokenizer \
  --train_file train_with_genre.json \
  --do_train \
  --output_dir gpt2-output-genre-boost-rag \
  --overwrite_output_dir \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 1 \
  --block_size 256 \
  --save_steps 500 \
  --evaluation_strategy no \
  --save_strategy no \
  --eval_steps 500 \
  --logging_steps 50 \
  --report_to wandb \
  --run_name "genre1" \
  --save_total_limit 2 \
  --load_best_model_at_end false \
  --fp16 \
  --fp16_full_eval \
  --eval_accumulation_steps 1
