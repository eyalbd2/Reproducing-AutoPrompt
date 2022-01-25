#!/bin/bash
# Run this command : bash run_manual_autoprompt.sh <GPU_ID> <MODEL_NAME>

# There is no importance of the chosen source domain, since the model does not perform any training here.
DOMAIN=airline


# Generating Prompts
CUDA_VISIBLE_DEVICES=$1 python -m autoprompt.pre_defined_prompt \
    --dev blitzer_data/${DOMAIN}/dev.tsv \
    --test 'blitzer_data' \
    --trg_domains 'airline,books,dvd,electronics,kitchen' \
    --template '<s> {sentence} . [T] [T] [P] . </s>' \
    --saved_label_map label_tokens/manual/general \
    --initial_trigger_path trigger_tokens/manual/general \
    --bsz 4 \
    --eval-size 8 \
    --model-name $2


