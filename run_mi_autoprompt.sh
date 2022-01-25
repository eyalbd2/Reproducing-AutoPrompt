#!/bin/bash
# Run this command : bash run_mi_autoprompt.sh <GPU_ID> <MODEL_NAME>

# '[CLS] {sentence} [T] [T] [T] [P]. [SEP]' (bert-base-cased)    or    <s> {sentence} [T] [T] [T] [P] . </s> (roberta-base)
DOMAIN=airline

# Label Token Selection
python -m autoprompt.get_mi_label_tokens \
    --pivot_num 3 \
    --src_threshold 20 \
    --trg_threshold 50 \
    --src ${DOMAIN}

python -m autoprompt.get_mi_triggers \
  --src_threshold 100 \
  --trg_threshold 200 \
  --src ${DOMAIN} \
  --label_token_path label_tokens/mi/${DOMAIN}


# Generating Prompts
CUDA_VISIBLE_DEVICES=$1 python -m autoprompt.pre_defined_prompt \
    --dev blitzer_data/${DOMAIN}/dev.tsv \
    --test 'blitzer_data' \
    --trg_domains 'airline,books,dvd,electronics,kitchen' \
    --template '<s> {sentence} [T] [T] [T] [P] . </s>' \
    --saved_label_map label_tokens/mi/${DOMAIN} \
    --initial_trigger_path trigger_tokens/mi/${DOMAIN} \
    --bsz 4 \
    --eval-size 8 \
    --model-name $2


