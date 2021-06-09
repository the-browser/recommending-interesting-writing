# recommending-interesting-writing

**Presented at INTRS @ RECSYS 2020 workshop**

Paper: http://ceur-ws.org/Vol-2682/short2.pdf

Talk: https://www.youtube.com/watch?t=8974&v=6uO2KwjCgXE

The process for this project is outlined below:

### 1. Collecting Data

Data was scraped from various online publications, including from publications focused on both feature writing and those who primarily do traditional journalism. The fake-news corpus was also utilized

### 2. Training

Training data was sent through training process in batches with an even number of positive and negative examples for each batch. Evaluation recall (percentage correctly labelled of top 100 predictions) was calculated every 25 steps for a variety of differently tuned model. Training was stopped after validation performance decreased for three successive iterations, at which point the state at which the model had the highest recall was used to generate results on the test dataset. The best model was then chosen in terms of both evaluation and test performance. Both BERT and rankfromsets were tested to compare the models in terms of efficency and ability to accurately give good recommendations. Rankfromsets proved faster and exhibited higher performance on both test and validation sets.

### 3. Lambda Deployment

To set up a proper website demo that could fetch articles, we needed a fast approach to generating predictions. We converted all datasets to matrices, including word-embeddings and words in each article, then utilized optimized matrix multiplication to generate new results based on user inputted values for publication embedding. We also fetched top and least contributing words to provide more detail about why articles were rated highly. In addition, topic sliders were included that upweighted words similar to common themes, such as crime or politics, when generating new predictions. This allows for better filtering of data as it pertains to user preference, and can serve as an effective mechanism to narrow down choices amongst various pieces for businesses trying to fill specific criterion.

### 4. GitHub Pages

The website was then deployed to github pages and you can access it here: https://the-browser.github.io/recommending-interesting-writing/.
The setup allows you to control the scalar value for editing topic weights. We also introduced a coronavirus tab that can help you discover more intriguing writing pertaining to the virus. Articles were scraped from various publications, and then stored in AWS S3 to allow the model to generate flexible rankings as part of the interactive web service.

### Model Configuration

All files required for model loading can be found in this folder: https://drive.google.com/drive/folders/1Hg8rO0ftINuMAOh3Ta40XdE6KWdW7r4I?usp=sharing

**rankfromsets**

To configure the rankfromsets model, simply use this structure:

```
kwargs = dict(
    n_publications=len(final_publication_ids),
    n_articles=len(final_url_ids),
    n_attributes=len(final_word_ids),
    emb_size=10,
    sparse=False,
    use_article_emb=False,
    mode='mean',
)
model = InnerProduct(**kwargs)
model.load_state_dict(torch.load(rankfromsets_model_path))
```

where final_word_ids, final_url_ids, and final_publication_ids map to dictionaries/word_dictionary.json, article_dictionary.json, and publication_dictionary.json respectively and the model file is listed as rankfromsets-emb_size=10.pt. All of these files/folders are within the rankfromsets directory.

**BERT**

To configure the best performing BERT model trained, use this layout with the HuggingFace Transformers library

```
config = BertConfig.from_json_file(config_file)
model = BertForSequenceClassification(config)
model.load_state_dict(torch.load(BERT_model_path))
```

where the config_file is listed as config.json and the model file is named best-BERT-model.pt, both of which are in the BERT folder.
