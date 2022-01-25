# Examining AutoPrompt's Out-of-Domain Performance for Sentiment Analysis

## AutoPrompt
An automated method based on gradient-guided search to create prompts for a diverse set of NLP tasks. AutoPrompt demonstrates that masked language models (MLMs) have an innate ability to perform sentiment analysis, natural language inference, fact retrieval, and relation extraction. Here is a reference to the official [website](https://ucinlp.github.io/autoprompt/).


 
In this repository we present code that reproduces autoprompt low-resource result on Sentiment Analysis, based on the original implementation published on the authors [GitHub](https://github.com/ucinlp/autoprompt). Furtheremore, we experiment autoprompt in challenging out-of-domain settings and compare it to alternative approaches, including manually-created prompts, mutual-information (MI) based label-tokens extraction, and fully MI-based approaches (trigger and label tokens extraction). 


Our code is implemented in [PyTorch](https://pytorch.org/), using the [Transformers](https://github.com/huggingface/transformers) libraries. 

## Usage Instructions

Running an experiment with AutoPrompt consists of the following steps:

1. Define a prompt template that is adapted to the experimented pretrained LM. 
2. Extract label tokens. 
3. Extract trigger tokens.
4. Test the algorithm on OOD data.

Make sure your virtual env includes all requirements (specified in 'autoprompt_env.yml').

### Setup a conda environment
You can run the following command to create a conda environment from our .yml file:
```
conda env create --file autoprompt_env.yml
conda activate autoprompt
```

Next, we go through these steps using our running example:
- Source domain - _airline_.
- Target domains - _books_, _dvd_, _electronics_, _kitchen_.
We use a specific set of hyperparameters, following the details presented in the original paper. 


Notice, you can run all the above steps with a single command by running one of the following scripts: `run_autoprompt.sh`, `run_autoprompt_with_mi_labels.sh`, `run_mi_autoprompt.sh`, and `run_manual_autoprompt.sh`. 

For example, you can run the following command:
```
bash run_autoprompt.sh <GPU_ID> roberta-base
```
This will run autoprompt on top of a pretrained RoBERTa-base model.

We next provide explanation to each of the algorithms tested with these scripts:

1. `run_autoprompt.sh`
Performs the original algorithm proposed by the official paper.

2. `run_autoprompt_with_mi_labels.sh`
This algorithm first chooses the label tokens according to mutual-information calculation between unigrams and the task labels. Then, the algorithm tunes the trigger tokens similarly to how autoprompt does.

3. `run_mi_autoprompt.sh`
First, label tokens are extracted according to mutual-information. Then, we search for tri-grams that obtain high mutual-information with the following label: 1- if the tri-gram is followed by one of the label tokens, and 0- otherwise. This is supposed to extract syntactic triggers, in contrast to autoprompt's algorithm.

4. `run_manual_autoprompt.sh`
Here, we manually design the both the label tokens and the trigger tokens.




