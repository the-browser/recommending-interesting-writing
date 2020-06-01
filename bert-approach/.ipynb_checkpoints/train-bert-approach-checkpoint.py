# import necessary libraries
import pandas as pd
import re
import torch
import collections
import numpy as np
import json
import time
import torch.nn as nn
import os
import argparse
import arguments.train_arguments as arguments
from data_processing.articles import Articles
from models.models import InnerProduct
import data_processing.dictionaries as dictionary
import sampling.sampler_util as sampler_util
import training.eval_util as eval_util
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from transformers import BertTokenizer

parser = argparse.ArgumentParser(description='Train model on article data and test evaluation')
arguments.add_data(parser)
arguments.add_training(parser)
arguments.add_model(parser)
arguments.add_optimization(parser)
args = parser.parse_args()

# set device
if torch.cuda.is_available() and args.use_gpu:
    device = "cuda"
elif not args.use_gpu:
    device = "cpu"
else:
    device = "cpu"
    print("Cannot use GPU. Using CPU instead.")
print(f"Device: {device}")

# set output directory path
output_path = Path(args.output_dir)

# tensboard log and graph output folder declaration
log_tensorboard_dir = output_path / "runs" / args.word_embedding_type
writer = SummaryWriter(log_tensorboard_dir)

# load datasets
train_path = Path(args.train_path)
test_path = Path(args.test_path)
eval_path = Path(args.eval_path)

train_data = Articles(train_path)
test_data = Articles(test_path)
eval_data = Articles(eval_path)
print("Data Loaded")

# initialize tokenizer from BERT library
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
print("Tokenizer Initialized!")

# check and tokenize data if items need to be tokenized
if args.map_items and args.tokenize:
    train_data.tokenize(tokenizer)
    print("Train Data Tokenized")
    test_data.tokenize(tokenizer)
    print("Test Data Tokenized")
    eval_data.tokenize(tokenizer)
print("All Data Tokenized!")

# create and save or load dictionaries based on arguments
if args.create_dicts:
    final_word_ids, final_url_ids, final_publication_ids = dictionary.create_merged_dictionaries(train_data.examples, tokenizer, "target")
    print("Dictionaries Created")

    dict_path = Path(args.data_dir) / "dictionaries"
    if not dict_path.is_dir():
        dict_path.mkdir()

    dictionary.save_dictionaries(final_word_ids, final_url_ids, final_publication_ids, dict_path)
else:
    dictionary_dir = Path(args.dict_dir)
    final_word_ids,final_url_ids, final_publication_ids = dictionary.load_dictionaries(dictionary_dir)
    print("Dictionaries loaded.")

# map items in dataset using dictionary keys (convert words and urls to numbers for the model)
if args.map_items:
    for dataset in [train_data, test_data, eval_data]:
        dataset.map_items(tokenizer,
                          final_url_ids,
                          final_publication_ids,
                          filter=False)
    print("Items mapped")
    mapped_data_path = Path(args.data_dir) / "mapped-data"
    if not mapped_data_path.is_dir():
        mapped_data_path.mkdir()

    train_mapped_path = mapped_data_path / "train.json"
    test_mapped_path = mapped_data_path / "test.json"
    eval_mapped_path = mapped_data_path / "evaluation.json"
    with open(train_mapped_path, "w") as file:
        json.dump(train_data.examples, file)
    with open(test_mapped_path, "w") as file:
        json.dump(test_data.examples, file)
    with open(eval_mapped_path, "w") as file:
        json.dump(eval_data.examples, file)
    print(f"Mapped Data saved to {mapped_data_path} directory")

# create weights for dataset samples to ensure only positive and negative examples are chosen in respective samples
pos_sampler = train_data.create_positive_sampler(args.target_publication)
neg_sampler = train_data.create_negative_sampler(args.target_publication)

train_batch_sampler = sampler_util.BatchSamplerWithNegativeSamples(
    pos_sampler=pos_sampler, neg_sampler=neg_sampler,
    items=train_data.examples, batch_size=args.batch_size)
# define function to return necessary data for dataloader to pass into model
def collate_fn(examples):
    words = []
    articles = []
    labels = []
    publications = []
    for example in examples:
        if args.use_all_words:
            words.append(list(set(example['text'])))
        else:
            if len(example['text']) > args.words_to_use:
                words.append(list(set(example['text'][:args.words_to_use])))
            else:
                print(list(set(example['text'])))
                words.append(list(set(example['text'])))
        articles.append(example['url'])
        publications.append(example['model_publication'])
        labels.append(example['model_publication'])
    num_words = [len(x) for x in words]
    words = np.concatenate(words, axis=0)
    word_attributes = torch.tensor(words, dtype=torch.long)
    articles = torch.tensor(articles, dtype=torch.long)
    num_words.insert(0, 0)
    num_words.pop(-1)
    attribute_offsets = torch.tensor(np.cumsum(num_words), dtype=torch.long)
    publications = torch.tensor(publications, dtype=torch.long)
    real_labels = torch.tensor(labels, dtype=torch.long)
    return publications, articles, word_attributes, attribute_offsets, real_labels


# change negative example publication ids to the ids of the first half for predictions
def collate_with_neg_fn(examples):
    publications, articles, word_attributes, attribute_offsets, real_labels = collate_fn(examples)
    publications[len(publications)//2:] = publications[:len(publications)//2]
    return publications, articles, word_attributes, attribute_offsets, real_labels

# pin memory if using GPU for high efficiency
if device == "cuda":
    pin_mem = True
else:
    pin_mem = False

# create dataloaders for iterable data when training and testing recall
train_loader = torch.utils.data.DataLoader(train_data, batch_sampler=train_batch_sampler, collate_fn=collate_with_neg_fn, pin_memory=pin_mem)
eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=len(eval_data), collate_fn=collate_fn, pin_memory=pin_mem)
test_loader = torch.utils.data.DataLoader(train_data, batch_size=len(test_data), collate_fn=collate_fn, pin_memory=pin_mem)

# function that allows for infinite iteration over training batches
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


# initialize model, loss, and optimizer
kwargs = dict(n_publications=len(final_publication_ids),
              n_articles=len(final_url_ids),
              n_attributes=len(final_word_ids),
              emb_size=args.emb_size, sparse=args.use_sparse,
              use_article_emb=args.use_article_emb,
              mode=args.word_embedding_type)
model = InnerProduct(**kwargs)
model.reset_parameters()
model.to(device)

loss = torch.nn.BCEWithLogitsLoss()

if args.optimizer_type == "RMS":
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.momentum)
else:
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum)

print(model)
print(optimizer)
model.train()  # turn on training mode
check = True
running_loss = 0

labels = torch.Tensor((np.arange(args.batch_size) < args.batch_size // 2).astype(np.float32))
labels = labels.to(device)

print("Beginning Training")
print("--------------------")
# training loop with validation checks every 100 steps and final validation recall calcualation
for step, batch in enumerate(cycle(train_loader)):
    optimizer.zero_grad()
    publications, articles, word_attributes, attribute_offsets, real_labels = batch
    publications = publications.to(device)
    articles = articles.to(device)
    word_attributes = word_attributes.to(device)
    attribute_offsets = attribute_offsets.to(device)
    logits = model(publications, articles, word_attributes, attribute_offsets)
    L = loss(logits, labels)
    L.backward()
    optimizer.step()
    running_loss += L.item()
    if step % 100 == 0 and step % args.training_steps != 0:
        writer.add_scalar('Loss/train', running_loss/100, step)
        print(f"Training Loss: {running_loss/100}")
        sorted_preds, indices = eval_util.calculate_predictions(eval_loader,
                                                                model, device,
                                                                args.target_publication,
                                                                version="Evaluation",
                                                                step=step, writer=writer,
                                                                check_recall=True)
        model.train()
        running_loss = 0.0
    if step != 0 and step % args.training_steps == 0:
        writer.add_scalar('Loss/train', running_loss/100, step)
        print(f"Training Loss: {running_loss/100}")
        print("Getting Final Evaluation Results")
        print("--------------------")
        sorted_preds, indices = eval_util.calculate_predictions(eval_loader, model, device,
                                                                args.target_publication,
                                                                version="Evaluation",
                                                                step=step, writer=writer,
                                                                check_recall=True)
        ranked_df = eval_util.create_ranked_results_list(final_word_ids,
                                                      args.word_embedding_type,
                                                      sorted_preds, indices,
                                                      eval_data)
        eval_util.save_ranked_df(output_path,
                                "evaluation",
                                 ranked_df,
                                 args.word_embedding_type)
        print(f"Ranked Data Saved to {output_path / 'results' / 'evaluation'}!")
        check = False
        break

# save model for easy future reloading
model_path = output_path / "model"
if not model_path.is_dir():
    model_path.mkdir()
model_string = args.word_embedding_type + "-inner-product-model.pt"
model_path = model_path / model_string
torch.save(model.state_dict(), model_path)
sorted_preds, indices = eval_util.calculate_predictions(test_loader,
                                                        model, device,
                                                        args.target_publication,
                                                        version="Test",
                                                        step=step, writer=writer,
                                                        check_recall=True)
writer.close()
ranked_df = eval_util.create_ranked_results_list(final_word_ids,
                                                args.word_embedding_type,
                                                sorted_preds, indices,
                                                test_data)
eval_util.save_ranked_df(output_path,
                        "test",
                        ranked_df,
                        args.word_embedding_type)
print(f"Ranked Data Saved to {output_path / 'results' / 'test'}!")
print("Done!")