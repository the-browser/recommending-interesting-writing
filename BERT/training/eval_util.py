import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import pathlib as Path
import os
import time
from datetime import datetime
import numpy as np


def create_ranked_results_list(final_word_ids, sorted_preds, indices, data):
    df = pd.DataFrame(columns=["title", "url", "publication", "date", "prediction"])
    ranked_indices = indices[::-1]
    predictions = sorted_preds[::-1]
    for i in range(0, 1500):
        example = data[ranked_indices[i]]
        prediction = predictions[i]
        title = example["title"]
        url = example["link"]
        publication = example["publication"]
        date = example["date"]
        df.loc[i] = [title, url, publication, date, prediction]
    return df


def save_ranked_df(output_path, version, df):
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
    result_path = version + "-BERT-top-1500.csv"
    eval_folder_path = evaluation_results_path / result_path
    df.to_csv(eval_folder_path, index=False)


@torch.no_grad()
def calculate_batched_predictions(batch, model, device, target):
    model.eval()
    word_attributes, attention_masks, word_subset_counts, real_labels = batch
    word_attributes = word_attributes.to(device)
    attention_masks = attention_masks.to(device)
    logits = model(word_attributes, attention_mask=attention_masks)[0]
    final_logits = np.squeeze(logits.cpu().numpy())
    return final_logits


@torch.no_grad()
def calculate_recall(
    dataset, indices, recall_value, target_publication, version, writer, step,
):
    rev_indices = indices[::-1]
    correct_10 = 0
    correct_big = 0
    for i in range(recall_value):
        if dataset[rev_indices[i]]["model_publication"] == target_publication:
            if i < 10:
                correct_10 += 1
            correct_big += 1
    print(f"{version} Performance: Step - {step}")
    print(f"Top 10: {correct_10} / 10 or {correct_10*10} %")
    print(
        f"Top {str(recall_value)}: {correct_big} / {str(recall_value)} or {(correct_big*100)/recall_value} %"
    )
    print("--------------------")
    writer.add_scalar(f"{version}/Top-10", correct_10, step)
    writer.add_scalar(f"{version}/Top-{recall_value}", correct_big, step)
    return correct_big
