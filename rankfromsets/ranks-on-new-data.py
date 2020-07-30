# import necessary libraries
import torch
import numpy as np
import ujson as json
import torch.nn as nn
import argparse
from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from transformers import BertForSequenceClassification, BertConfig
from tqdm import tqdm
import numpy as np
import arguments.rank_arguments as arguments
from data_processing.articles import Articles
import data_processing.dictionaries as dictionary
import training.eval_util as eval_util
from training.collate import collate_fn

parser = argparse.ArgumentParser(description="Get Ranked Predictions on New Dataset.")
arguments.add_data(parser)
arguments.add_model(parser)
args = parser.parse_args()

if torch.cuda.is_available() and args.use_gpu:
    device = "cuda"
elif not args.use_gpu:
    device = "cpu"
else:
    device = "cpu"
    print("Cannot use GPU. Using CPU instead.")
print(f"Device: {device}")
print("-------------------")

# set output directory path
output_path = Path(args.output_dir)

# load in dataset
raw_data_path = Path(args.dataset_path)
raw_data = Articles(raw_data_path)
print("Data Loaded")
print("-------------------")

# load dictionaries from path
dictionary_dir = Path(args.dict_dir)
final_word_ids, final_url_ids, final_publication_ids = dictionary.load_dictionaries(
    dictionary_dir
)
print("Dictionaries Loaded")
print("-------------------")


# map items to their dictionary values
if args.map_items:
    # initialize tokenizer from BERT library
    tokenizer = BertWordPieceTokenizer(args.tokenizer_file, lowercase=True)
    print("Tokenizer Initialized!")

    # tokenize and map items to their ids in dictionaries and filter articles
    proper_data = raw_data.map_items(
        tokenizer,
        final_word_ids,
        final_url_ids,
        final_publication_ids,
        filter=True,
        min_length=args.min_article_length,
        day_range=args.days_old,
    )
    print("Mapped and Filtered Data!")
    data_path = Path(args.data_dir)
    if not data_path.is_dir():
        data_path.mkdir()
    mapped_data_path = data_path / "mapped-data"
    print("Initial: ", len(raw_data))
    if not mapped_data_path.is_dir():
        mapped_data_path.mkdir()
    train_mapped_path = mapped_data_path / "mapped_dataset.json"
    with open(train_mapped_path, "w") as file:
        json.dump(proper_data, file)
    raw_data = Articles(train_mapped_path)
    print("Final: ", len(raw_data))
    print(f"Filtered, Mapped Data saved to {mapped_data_path} directory")
    print("-------------------")


def collate_fn(examples):
    words = []
    articles = []
    labels = []
    publications = []
    for example in examples:
        if args.use_all_words:
            words.append(list(set(example["text"])))
        else:
            if len(example["text"]) > args.words_to_use:
                words.append(list(set(example["text"][: args.words_to_use])))
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


# Generates a dataloader on the dataset that outputs entire set as a batch for one time predictions
raw_loader = torch.utils.data.DataLoader(
    raw_data, batch_size=args.data_batch_size, collate_fn=collate_fn, pin_memory=pin_mem
)

abs_model_path = Path(args.model_path)
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
print("Model Loaded")
print(model)
print("-------------------")

# get final evaluation results and create a basic csv of top articles
data_logit_list = []
for batch in tqdm(raw_data):
    current_logits = eval_util.calculate_batched_predictions(
        batch, model, device, args.target_publication
    )
    data_logit_list = data_logit_list + list(current_logits)
converted_list = np.array(eval_logit_list)
sorted_preds = np.sort(converted_list)
indices = np.argsort(converted_list)

ranked_df = eval_util.create_ranked_results_list(
    final_word_ids, sorted_preds, indices, raw_data
)
eval_util.save_ranked_df(output_path, "evaluation", ranked_df, args.word_embedding_type)
print("Predictions Made")
print(f"Ranked Data Saved to {output_path / 'results' / 'evaluation'} directory!")
