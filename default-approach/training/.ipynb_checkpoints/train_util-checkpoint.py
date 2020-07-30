import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import pathlib as Path
import os


#create evaluation batch and send to device
def create_eval_batch(eval_loader, device):
    eval_batch = next(iter(eval_loader))
    eval_publications, eval_articles, eval_word_attributes, eval_attribute_offsets, eval_real_labels = eval_batch
    eval_articles = eval_articles.to(device)
    eval_word_attributes = eval_word_attributes.to(device)
    eval_attribute_offsets = eval_attribute_offsets.to(device)
    eval_real_labels = eval_real_labels.to(device)
    return eval_publications, eval_articles, eval_word_attributes, eval_attribute_offsets, eval_real_labels


def calculate_eval_performance(eval_loader, model, device, target_publication, step, writer):
    eval_publications, eval_articles, eval_word_attributes, eval_attribute_offsets, eval_real_labels = create_eval_batch(eval_loader, device)
    model.eval()
    publication_set = [target_publication]*len(eval_real_labels)
    publication_set = torch.tensor(publication_set, dtype=torch.long)
    publication_set = publication_set.to(device)
    preds = model(publication_set, eval_articles, eval_word_attributes, eval_attribute_offsets)
    sorted_preds, indices = torch.sort(preds, descending=True)
    correct_10 = 0
    correct_100 = 0
    for i in range(0, 100):
        if eval_real_labels[indices[i]] == target_publication:
            if i < 10 :
                correct_10 += 1
            correct_100 += 1
    print(f"Evaluation Performance: Step - {step}")
    print(f"Top 10: {correct_10} / 10 or {correct_10*10} %")
    print(f"Top 100: {correct_100} / 100 or {correct_100} %")
    print("--------------------")
    writer.add_scalar('Eval/Top-10', correct_10, step)
    writer.add_scalar('Eval/Top-100', correct_100, step)
    
        
def save_ranked_eval_list(final_word_ids, final_url_ids, output_path, word_embedding_type):
    df = pd.DataFrame(columns=['title', 'url', 'text',
                               'publication', 'target_prediction'])
    for i in range(0, 1500):
        example = eval_data[indices[i]]
        prediction = sorted_preds[i].item()
        text = []
        for x in example['title']:
            text.append(next((word for word, numero in final_word_ids.items() if numero == x), None))
            title = ""
        for word in text:
            title += word
            title += " "
        unique_text = list(set(example['text']))
        url = next((url for url, numero in final_url_ids.items() if numero == example['url']), None)
        publication = example['publication']
        df.loc[i] = [title, url, unique_text, publication, prediction]
    results_path = output_path / "results"
    if not results_path.is_dir():
        results_path.mkdir()
    evaluation_results_path = results_path / "evaluation"
    if not evaluation_results_path.is_dir():
        evaluation_results_path.mkdir()
    result_path = word_embedding_type + "-top-1500.csv"
    eval_folder_path = evaluation_results_path / result_path
    df.to_csv(eval_folder_path, index=False)