#!/bin/bash
# Run this command : bash run_sentiment_sst2.sh <GPU_ID> <MODEL_NAME>

# '[CLS] {sentence} [T] [T] [T] [P]. [SEP]' (bert-base-cased)    or    <s> {sentence} [T] [T] [T] [P] . </s> (roberta-base)
# {"0": ["Ġworse", "Ġincompetence", "ĠWorse", "Ġblamed", "Ġsucked"], "1": ["ĠCris", "Ġmarvelous", "Ġphilanthrop", "Ġvisionary", "Ġwonderful"]}

# Label Token Selection
#CUDA_VISIBLE_DEVICES=$1 python -m autoprompt.label_search \
#    --train blitzer_data/airline/train.tsv \
#    --template '<s> {sentence} [T] [T] [T] [P] . </s>' \
#    --label-map '{"0": 0, "1": 1}' \
#    --iters 30 \
#    --model-name $2 #'bert-base-cased', 'roberta-base'


# Generating Prompts
CUDA_VISIBLE_DEVICES=$1 python -m autoprompt.create_trigger \
    --train blitzer_data/airline/train.tsv \
    --dev blitzer_data/airline/dev.tsv \
    --test blitzer_data/dvd/test-labeled.tsv \
    --template '<s> {sentence} [T] [T] [T] [P] . </s>' \
    --label-map '{"0": ["Ġrapist", "Ġbarbaric", "ĠSahara"], "1": ["Ġshone", "ooth", "Ġshine"]}' \
    --num-cand 100 \
    --accumulation-steps 10 \
    --bsz 4 \
    --eval-size 8 \
    --iters 40 \
    --model-name $2 #roberta-large
