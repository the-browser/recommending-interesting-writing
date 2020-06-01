import json
import numpy as np
import boto3
import scipy
import scipy.sparse
from io import BytesIO
import os
import base64

ACCESS_KEY = os.environ['ACCESS_KEY']
SECRET_ACCESS_KEY = os.environ['SECRET_ACCESS_KEY']

def getData(dataset):
    BUCKET = 'personal-bucket-news-ranking'
    client = boto3.client('s3',
                          aws_access_key_id=ACCESS_KEY,
                          aws_secret_access_key=SECRET_ACCESS_KEY
                          )
    path_to_data = "sets/csr_articles_" + dataset +".npz"
    result = client.get_object(Bucket=BUCKET, Key=path_to_data)
    word_articles = scipy.sparse.load_npz(BytesIO(result["Body"].read()))

    FILE_TO_READ = 'word_emb.npy'
    result = client.get_object(Bucket=BUCKET, Key=FILE_TO_READ)
    word_emb = np.load(BytesIO(result["Body"].read()))

    FILE_TO_READ = 'word_bias.npy'
    result = client.get_object(Bucket=BUCKET, Key=FILE_TO_READ)
    word_bias = np.load(BytesIO(result["Body"].read()))

    FILE_TO_READ = 'reversed_word_ids.json'
    result = client.get_object(Bucket=BUCKET, Key=FILE_TO_READ)
    id_to_word = json.loads(result["Body"].read().decode())

    FILE_TO_READ = 'sets/mapped_dataset_' + dataset + '.json'
    result = client.get_object(Bucket=BUCKET, Key=FILE_TO_READ)
    real_data = json.loads(result["Body"].read().decode())

    return word_articles, word_emb, word_bias, id_to_word, real_data


def lambda_handler(event, context):
    publication_emb = np.asarray([
        1.9955785,
        2.0590432,
        2.1761959,
        -2.0220742,
        1.9912238,
        -2.027111,
        -2.2134302,
        -1.9901487,
        2.0043654,
        -2.3093524,
        -2.0107682,
        2.002998,
        2.0343666,
        -2.0904067,
        2.4320478,
        2.0393133,
        2.0141761,
        1.992825,
        1.9798595,
        -2.0268648,
        -1.9863509,
        2.0398421,
        -1.9834707,
        -2.0138183,
        1.997613,
    ], dtype=np.float32)

    publication_bias = 0.45260596
    emb_dict = json.loads(event['body'])
    
    publication_emb[emb_dict['idx1']] = emb_dict['a']
    publication_emb[emb_dict['idx2']] = emb_dict['b']
    publication_emb[emb_dict['idx3']] = emb_dict['c']
    publication_emb[emb_dict['idx4']] = emb_dict['d']
    word_articles, word_emb, word_bias, id_to_word, real_data = getData(emb_dict['dataset'])
    print("Data loaded successfully!")

    article_embeddings = word_articles.dot(word_emb)

    emb_times_publication = np.dot(article_embeddings, publication_emb.reshape(25,1))

    article_bias = word_articles.dot(word_bias)

    product_with_bias = emb_times_publication + article_bias

    word_counts = word_articles.sum(axis=1).reshape(word_articles.shape[0], 1)

    final_logits = np.divide(product_with_bias, word_counts) + float(publication_bias)

    indices = final_logits.argsort(axis=0)[-75:].reshape(75)

    word_logits = np.dot(word_emb, publication_emb.reshape(25,1)) + word_bias

    top_articles = word_articles[indices.tolist()[0]]

    broadcasted_words_per_article = top_articles.toarray() * word_logits.T

    sorted_word_indices = broadcasted_words_per_article.argsort(axis=1)

    return_articles = []

    i = 0
    for idx in indices.tolist()[0]:
        current_article = real_data[int(idx)]
        current_article['logit'] = float(final_logits[int(idx)])
        current_sorted_words = sorted_word_indices[i]
        top_words = []
        least_words = []
        for top_word in current_sorted_words[-20:]:
            word = id_to_word[str(top_word)]
            top_words.append(word)
        for least_word in current_sorted_words[:20]:
            word = id_to_word[str(least_word)]
            least_words.append(word)
        current_article['top_words'] = top_words
        current_article['least_words'] = least_words
        return_articles.append(current_article)
        i += 1
    ordered_return_articles = return_articles[::-1]
    origin = event['headers']['origin']
    if origin in ['https://the-browser.github.io', 'https://rohanbansal12.github.io']:
        response = {
            "statusCode": 200,
            "body": json.dumps(ordered_return_articles),
            'headers': {
                'Access-Control-Allow-Origin': origin,
            },
        }
        return response
    else:
        response = {
            "statusCode": 500,
        }
        return response
