# import necessary libraries
import pandas as pd
import re
import torch
import collections
import numpy as np
import ujson as json
import time
import torch.nn as nn
import os
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tokenizers import BertWordPieceTokenizer
import random
from tqdm import tqdm
import numpy as np
import arguments.train_arguments as arguments
from data_processing.articles import Articles
from models.models import InnerProduct
import data_processing.dictionaries as dictionary
import sampling.sampler_util as sampler_util
import training.eval_util as eval_util

np.random.seed(0)

parser = argparse.ArgumentParser(
    description="Train model on article data and test evaluation"
)
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
eval_data = Articles(eval_path, index_file=args.index_file_path)
print("Data Loaded")

# initialize tokenizer from BERT library
tokenizer = BertWordPieceTokenizer(args.tokenizer_file, lowercase=True)
print("Tokenizer Initialized!")

# create and save or load dictionaries based on arguments
if args.create_dicts:
    (
        final_word_ids,
        final_url_ids,
        final_publication_ids,
    ) = dictionary.create_merged_dictionaries(
        train_data.examples, "target", args.tokenizer_file
    )
    print("Dictionaries Created")

    dict_path = Path(args.data_dir) / "dictionaries"
    if not dict_path.is_dir():
        dict_path.mkdir()

    dictionary.save_dictionaries(
        final_word_ids, final_url_ids, final_publication_ids, dict_path
    )
else:
    dictionary_dir = Path(args.dict_dir)
    final_word_ids, final_url_ids, final_publication_ids = dictionary.load_dictionaries(
        dictionary_dir
    )
    print("Dictionaries loaded.")

# map items in dataset using dictionary keys (convert words and urls to numbers for the model)
if args.map_items:
    badtokens = []
    if args.bad_token_path.is_file():
        bad_token_path = Path(args.bad_token_path)
        with open(bad_token_path, "r") as f:
            badTokens = [int(line.rstrip()) for line in f]

        for dataset in [train_data, test_data, eval_data]:
            dataset.map_items(
                tokenizer, final_url_ids, final_publication_ids, filter=False,
            )
    else:
        for dataset in [train_data, test_data, eval_data]:
            dataset.map_items(
                tokenizer, final_url_ids, final_publication_ids, filter=False
            )
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
    pos_sampler=pos_sampler,
    neg_sampler=neg_sampler,
    items=train_data.examples,
    batch_size=args.batch_size,
)


# define function to return necessary data for dataloader to pass into model
def collate_fn(examples):
    words = []
    articles = []
    labels = []
    publications = []
    for example in examples:
        if args.use_all_words:
            words.append(list(set(example["text"])))
        else:
            if len(list(set(example["text"]))) > args.words_to_use:
                words.append(
                    random.sample(list(set(example["text"])), args.words_to_use)
                )
            else:
                words.append(list(set(example["text"])))
        articles.append(example["url"])
        publications.append(example["model_publication"])
        labels.append(example["model_publication"])
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
    (
        publications,
        articles,
        word_attributes,
        attribute_offsets,
        real_labels,
    ) = collate_fn(examples)
    publications[len(publications) // 2 :] = publications[: len(publications) // 2]
    return publications, articles, word_attributes, attribute_offsets, real_labels


# pin memory if using GPU for high efficiency
if args.use_gpu:
    pin_mem = True
else:
    pin_mem = False

# create dataloaders for iterable data when training and testing recall
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_sampler=train_batch_sampler,
    collate_fn=collate_with_neg_fn,
    pin_memory=pin_mem,
)
eval_loader = torch.utils.data.DataLoader(
    eval_data, batch_size=10000, collate_fn=collate_fn, pin_memory=pin_mem
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=10000, collate_fn=collate_fn, pin_memory=pin_mem
)


# function that allows for infinite iteration over training batches
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


# initialize model, loss, and optimizer
kwargs = dict(
    n_publications=len(final_publication_ids),
    n_articles=len(final_url_ids),
    n_attributes=len(final_word_ids),
    emb_size=args.emb_size,
    sparse=args.use_sparse,
    use_article_emb=args.use_article_emb,
    mode=args.word_embedding_type,
)
model = InnerProduct(**kwargs)
model.reset_parameters()
pre_stuff = np.loadtxt(
    "/scratch/gpfs/altosaar/dat/longform-data/BERT/word_embs.tsv", delimiter="\t"
)
model.attribute_emb_sum.weight.data = torch.tensor(pre_stuff)
model.to(device)

loss = torch.nn.BCEWithLogitsLoss()

if args.optimizer_type == "RMS":
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )
else:
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )

print(model)
print(optimizer)
model.train()  # turn on training mode
check = True
running_loss = 0

labels = torch.Tensor(
    (np.arange(args.batch_size) < args.batch_size // 2).astype(np.float32)
)
labels = labels.to(device)

validation_recall_list = []
print("Beginning Training")
print("--------------------")

# training loop with validation checks
for step, batch in enumerate(cycle(train_loader)):
    # calculate test and evaluation performance based on user intended frequency
    if step % args.frequency == 0 and step != args.training_steps:
        # output loss
        model.eval()
        torch.no_grad()
        writer.add_scalar("Loss/train", running_loss / args.frequency, step)
        print(f"Training Loss: {running_loss/args.frequency}")
        running_loss = 0.0
        logit_list = []
        for eval_batch in tqdm(eval_loader):
            current_logits = eval_util.calculate_batched_predictions(
                eval_batch, model, device, args.target_publication
            )
            logit_list = logit_list + list(current_logits)
        converted_list = np.array(logit_list)
        sorted_preds = np.sort(converted_list)
        indices = np.argsort(converted_list)
        calc_recall = eval_util.calculate_recall(
            eval_data,
            indices,
            args.eval_recall_max,
            args.target_publication,
            "Eval",
            writer,
            step,
        )
        validation_recall_list.append(calc_recall)

        # save model for easy reloading
        if max(validation_recall_list) == validation_recall_list[-1]:
            model_path = output_path / "model"
            if not model_path.is_dir():
                model_path.mkdir()
            model_string = (
                str(step) + args.word_embedding_type + "-inner-product-model.pt"
            )
            model_path = model_path / model_string
            torch.save(model.state_dict(), model_path)

        # check if validation recall is increasing
        if len(validation_recall_list) > 3:
            full_length = len(validation_recall_list)
            if (
                validation_recall_list[-1] < validation_recall_list[-2]
                and validation_recall_list[-2] < validation_recall_list[-3]
                and validation_recall_list[-3] < validation_recall_list[-4]
            ):
                print("Validation Recall Decreased For Two Successive Iterations!")
                break

    # turn to training mode and calculate loss for backpropagation
    torch.enable_grad()
    model.train()
    optimizer.zero_grad()
    publications, articles, word_attributes, attribute_offsets, real_labels = batch
    publication_set = [args.target_publication] * len(real_labels)
    publication_set = torch.tensor(publication_set, dtype=torch.long)
    publication_set = publication_set.to(device)
    articles = articles.to(device)
    word_attributes = word_attributes.to(device)
    attribute_offsets = attribute_offsets.to(device)
    logits = model(publication_set, articles, word_attributes, attribute_offsets)
    L = loss(logits, labels)
    L.backward()
    optimizer.step()
    running_loss += L.item()
    if step != 0 and step % args.training_steps == 0:
        writer.add_scalar("Loss/train", running_loss / args.frequency, step)
        print(f"Training Loss: {running_loss/100}")
        print("Getting Final Evaluation Results")
        print("--------------------")
        break

# load best model for final performance metrics and data saving
proper_step_model = (
    str(np.argmax(validation_recall_list) * args.frequency)
    + args.word_embedding_type
    + "-inner-product-model.pt"
)

writer.add_scalar("Peaked_Steps", np.argmax(validation_recall_list) * args.frequency)
writer.add_scalar("Max_Evaluation_Recall", np.max(validation_recall_list))

abs_model_path = output_path / "model" / proper_step_model
kwargs = dict(
    n_publications=len(final_publication_ids),
    n_articles=len(final_url_ids),
    n_attributes=len(final_word_ids),
    emb_size=args.emb_size,
    sparse=args.use_sparse,
    use_article_emb=args.use_article_emb,
    mode=args.word_embedding_type,
)
model = InnerProduct(**kwargs)
model.load_state_dict(torch.load(abs_model_path))
model.to(device)
model.eval()
torch.no_grad()

# get final evaluation results and create a basic csv of top articles
eval_logit_list = []
for batch in tqdm(eval_loader):
    current_logits = eval_util.calculate_batched_predictions(
        batch, model, device, args.target_publication
    )
    eval_logit_list = eval_logit_list + list(current_logits)
converted_list = np.array(eval_logit_list)
sorted_preds = np.sort(converted_list)
indices = np.argsort(converted_list)
calc_recall = eval_util.calculate_recall(
    eval_data,
    indices,
    args.eval_recall_max,
    args.target_publication,
    "Eval",
    writer,
    step,
)
ranked_df = eval_util.create_ranked_results_list(
    final_word_ids, sorted_preds, indices, eval_data
)
eval_util.save_ranked_df(output_path, "evaluation", ranked_df, args.word_embedding_type)

# get final test results and create a basic csv of top articles
test_logit_list = []
for batch in tqdm(test_loader):
    current_logits = eval_util.calculate_batched_predictions(
        batch, model, device, args.target_publication
    )
    test_logit_list = test_logit_list + list(current_logits)
converted_list = np.array(test_logit_list)
sorted_preds = np.sort(converted_list)
indices = np.argsort(converted_list)
calc_recall = eval_util.calculate_recall(
    test_data,
    indices,
    args.test_recall_max,
    args.target_publication,
    "Test",
    writer,
    step,
)
ranked_df = eval_util.create_ranked_results_list(
    final_word_ids, sorted_preds, indices, test_data
)
eval_util.save_ranked_df(output_path, "test", ranked_df, args.word_embedding_type)

# close writer and exit
writer.close()
print(f"Ranked Data Saved to {output_path / 'results'}!")
print("Done!")
