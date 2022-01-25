
import argparse
import logging
from tqdm import tqdm, trange
import numpy as np
import torch
from torch import nn, cuda
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import transformers
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer,  BertConfig, BertModel, BertTokenizer
import logging
logging.basicConfig(level=logging.ERROR)


class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.sentence
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            self.text[index],
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class Classifier(torch.nn.Module):
    def __init__(self, num_labels, model_obj, model_name="roberta-base"):
        super(Classifier, self).__init__()
        self.num_labels = num_labels
        self.l1 = model_obj.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(768, self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # hidden_state = output_1[0]
        # pooler = hidden_state[:, 0]
        pooler = output_1[1]
        output = self.classifier(pooler)
        return output


def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct


def valid(model, dataloader):
    model.eval()
    n_correct, n_wrong, total, tr_loss, nb_tr_steps, nb_tr_examples = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(dataloader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

    epoch_accu = (n_correct * 100) / nb_tr_examples
    return epoch_accu


def train(epoch, train_dataloader, dev_dataloader):
    tr_loss, n_correct, nb_tr_steps, nb_tr_examples = 0, 0, 0, 0
    model.train()
    for _, data in tqdm(enumerate(train_dataloader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    train_loss = tr_loss / nb_tr_steps
    train_accu = (n_correct * 100) / nb_tr_examples
    dev_accu = valid(model, dev_dataloader)
    print(f'Epoch {epoch}:')
    print(f"    Train: Accuracy = {train_accu}, Loss = {train_loss}")
    print(f"    Dev:   Accuracy = {dev_accu}")
    return dev_accu


device = 'cuda' if cuda.is_available() else 'cpu'
MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-05
model_name = "bert-base-uncased"
task_name = "sentiment-blitzer"
freeze_encoder = False

modeling_dict = \
    {
        "roberta-base": (RobertaConfig(), RobertaModel(RobertaConfig()),
                         RobertaTokenizer.from_pretrained("roberta-base", truncation=True, do_lower_case=True)),
        "bert-base-uncased": (BertConfig(), BertModel(BertConfig()),
                              BertTokenizer.from_pretrained("bert-base-uncased", truncation=True, do_lower_case=True))
    }
task_dict = \
    {
        "sentiment-blitzer": (2, SentimentData)
    }
(configuration, model_obj, tokenizer) = modeling_dict[model_name]
(num_labels, task_processor) = task_dict[task_name]

train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
eval_params = {'batch_size': EVAL_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
all_domains = ["airline", "books", "dvd", "electronics", "kitchen"]
for source in ["sst2"]: #all_domains:
    all_targets = [domain for domain in all_domains if domain != source]

    # train_df = pd.read_csv(f'blitzer_data/{source}/train.tsv', delimiter='\t')
    train_df = pd.read_csv('glue_data/SST-2/train.tsv', delimiter='\t')
    print(f"TRAIN Dataset ({source}): {train_df.shape}")
    train_dataset = task_processor(train_df, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, **train_params)

    # dev_df = pd.read_csv(f'blitzer_data/{source}/dev.tsv', delimiter='\t')
    dev_df = pd.read_csv('glue_data/SST-2/dev.tsv', delimiter='\t')
    print(f"DEV Dataset ({source}): {dev_df.shape}")
    dev_dataset = task_processor(dev_df, tokenizer, MAX_LEN)
    dev_loader = DataLoader(dev_dataset, **eval_params)

    all_test_loaders = []
    for target in all_targets:
        test_df = pd.read_csv(f'blitzer_data/{target}/test-labeled.tsv', delimiter='\t')
        print(f"TEST Dataset ({target}): {test_df.shape}")
        test_dataset = task_processor(test_df, tokenizer, MAX_LEN)
        test_loader = DataLoader(test_dataset, **eval_params)
        all_test_loaders.append((target, test_loader))

    model = Classifier(num_labels=2, model_obj=model_obj, model_name=model_name)
    model.to(device)

    # Prepare optimizer
    if freeze_encoder:
        # freeze all bert weights, train only last encoder layer
        for param in model.l1.parameters():
            param.requires_grad = False

    # Creating the loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    best_dev_acc, final_test_acc = 0, {}
    for epoch in range(EPOCHS):
        dev_acc = train(epoch, train_loader, dev_loader)
        if dev_acc >= best_dev_acc:
            best_dev_acc = dev_acc
            print(f"Dev accuracy improved, collecting test results.")
            for (target_domain, test_dataloader) in all_test_loaders:
                test_accu = valid(model, test_dataloader)
                print(f"    {target_domain} = {test_accu}")
                final_test_acc[target_domain] = test_accu

    print(f"----------------------------------------")
    print(f"Final results:")
    print(f"Dev = {best_dev_acc}")
    for target in final_test_acc:
        print(f"{target} = {final_test_acc[target]}")
    print(f"----------------------------------------")

