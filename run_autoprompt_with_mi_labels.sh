#!/bin/bash
# Run this command : bash run_autoprompt_with_mi_labels.sh <GPU_ID> <MODEL_NAME>

# '[CLS] {sentence} [T] [T] [T] [P]. [SEP]' (bert-base-cased)    or    <s> {sentence} [T] [T] [T] [P] . </s> (roberta-base)
DOMAIN=kitchen

# Label Token Selection
python -m autoprompt.get_mi_label_tokens \
    --pivot_num 3 \
    --src_threshold 20 \
    --trg_threshold 50 \
    --src ${DOMAIN}


# Generating Prompts
CUDA_VISIBLE_DEVICES=$1 python -m autoprompt.create_trigger \
    --train blitzer_data/${DOMAIN}/train.tsv \
    --dev blitzer_data/${DOMAIN}/dev.tsv \
    --test 'blitzer_data' \
    --trg_domains 'airline,books,dvd,electronics,kitchen' \
    --template '<s> {sentence} [T] [T] [T] [P] . </s>' \
    --saved_label_map label_tokens/mi/${DOMAIN} \
    --num-cand 100 \
    --accumulation-steps 10 \
    --bsz 4 \
    --eval-size 8 \
    --iters 40 \
    --model-name $2


