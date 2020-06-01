import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import pathlib as Path
import os
import time
from datetime import datetime


#create a full batch and send to device
def create_full_batch(data_loader, device):
    data_batch = next(iter(data_loader))
    data_publications, data_articles, data_word_attributes, data_attribute_offsets, data_real_labels = data_batch
    data_articles = data_articles.to(device)
    data_word_attributes = data_word_attributes.to(device)
    data_attribute_offsets = data_attribute_offsets.to(device)
    data_real_labels = data_real_labels.to(device)
    return data_publications, data_articles, data_word_attributes, data_attribute_offsets, data_real_labels


def calculate_predictions(loader, model, device, target_publication, version='', step=0, check_recall=False, writer=None):
    publications, articles, word_attributes, attribute_offsets, real_labels = create_full_batch(loader, device)
    model.eval()
    publication_set = [target_publication]*len(real_labels)
    publication_set = torch.tensor(publication_set, dtype=torch.long)
    publication_set = publication_set.to(device)
    preds = model(publication_set, articles, word_attributes, attribute_offsets)
    sorted_preds, indices = torch.sort(preds, descending=True)
    if check_recall:
        correct_10 = 0
        correct_150 = 0
        for i in range(0, 150):
            if real_labels[indices[i]] == target_publication:
                if i < 10:
                    correct_10 += 1
                correct_150 += 1
        print(f"{version} Performance: Step - {step}")
        print(f"Top 10: {correct_10} / 10 or {correct_10*10} %")
        print(f"Top 150: {correct_150} / 150 or {(correct_150*2)/3} %")
        print("--------------------")
        if writer is not None:
            writer.add_scalar(f'{version}/Top-10', correct_10, step)
            writer.add_scalar(f'{version}/Top-150', correct_150, step)
    return sorted_preds, indices


def create_ranked_results_list(final_word_ids, word_embedding_type, sorted_preds, indices, data):
    df = pd.DataFrame(columns=['title', 'url', 'text',
                               'publication', 'target_prediction'])
    for i in range(0, 1500):
        example = data[indices[i]]
        prediction = sorted_preds[i].item()
        title = example['title']
        unique_text = list(set(example['text']))
        url = example['link']
        publication = example['publication']
        df.loc[i] = [title, url, unique_text, publication, prediction]
    return df


def save_ranked_df(output_path, version, df, word_embedding_type, word_count=0):
    if not output_path.is_dir():
        output_path.mkdir()
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%Y-%m-%d")
    results_path = output_path / "results"
    if not results_path.is_dir():
        results_path.mkdir()
    results_date_path = results_path / timestampStr
    if not results_date_path.is_dir():
        results_date_path.mkdir()
    evaluation_results_path = results_date_path / version
    if not evaluation_results_path.is_dir():
        evaluation_results_path.mkdir()
    if word_count != 0:
        result_path = "min-" + str(word_count) + "-" + word_embedding_type + "-top-1500.csv"
    else:
        result_path = word_embedding_type + "-top-1500.csv"
    eval_folder_path = evaluation_results_path / result_path
    df.to_csv(eval_folder_path, index=False)
