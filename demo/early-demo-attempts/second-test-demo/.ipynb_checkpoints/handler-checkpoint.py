import json
import numpy as np
import boto3
import scipy
import scipy.sparse
from io import BytesIO
import os

ACCESS_KEY = os.environ['ACCESS_KEY']
SECRET_ACCESS_KEY = os.environ['SECRET_ACCESS_KEY']


def getData():
    BUCKET = 'personal-bucket-news-ranking'
    client = boto3.client('s3',
                          aws_access_key_id=ACCESS_KEY,
                          aws_secret_access_key=SECRET_ACCESS_KEY
                          )
    FILE_TO_READ = 'csr_articles.npz'
    result = client.get_object(Bucket=BUCKET, Key=FILE_TO_READ)
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

    FILE_TO_READ = 'mapped_dataset.json'
    result = client.get_object(Bucket=BUCKET, Key=FILE_TO_READ)
    real_data = json.loads(result["Body"].read().decode())

    return word_articles, word_emb, word_bias, id_to_word, real_data


def lambda_handler(event, context):
    print(ACCESS_KEY)
    publication_emb = np.asarray([1.0440499, 1.0030843, 1.0340449, 0.992087, 1.0509816,
                                  1.0315005, -1.0493797, -1.0198538, 0.9712321, -1.026394,
                                 -0.9687971, 1.0592866, -1.0200703, -1.0423145, 0.9929519,
                                 1.0220934, 1.021279, -1.0265925, 0.9601833, 0.9763889,
                                 1.0109168, -0.9728226, 0.97199583, -1.0237931, -0.9996001,
                                 0.9932069, 0.97966635, -0.98893607, -0.9876815, -0.98812914,
                                 -0.9625895, 0.99879754, 0.9876508, -0.9581506, -0.95436096,
                                 -0.9601925, -1.0134513, -0.98763955, 0.98665, -1.0140482,
                                 1.004904, 0.9894275, -1.0044671, -0.9839679, -0.97082543,
                                 -0.9798079, 0.9926766, -0.97317344, 0.9797, -0.97642475,
                                 -0.99420726, -0.9972062, -1.0104703, 1.0575777, 0.9957696,
                                 -1.0413874, -1.0056863, -1.0151271, -0.99969465, 0.97463423,
                                 -0.98398715, -1.0211866, -1.0128828, -1.0024365, -0.9800189,
                                 1.0457181, 1.0155835, -1.036794, -1.013707, -1.0498024,
                                 -1.0252678, -1.0388161, -0.97501564, 0.97687274, 0.97906756,
                                 1.0536852, 1.0590494, -0.96917725, 1.0247189, -0.9818878,
                                 -1.0417286, -1.0204054, -1.0285249, -1.0329671, 0.9705739,
                                 0.96375024, 0.9891868, 0.9892464, 1.039075, 1.0042666,
                                 0.9786834, 1.0199072, 0.98080486, 0.9698635, -0.99322844,
                                 -0.95841753, -0.99150276, 0.97394156, 0.9976019, -1.0375009],
                                 dtype=np.float32)

    publication_bias = 0.99557
    publication_emb[1] = event['a']
    publication_emb[5] = event['b']
    publication_emb[17] = event['c']
    publication_emb[34] = event['d']
    publication_emb[67] = event['e']
    print(publication_emb)

    word_articles, word_emb, word_bias, id_to_word, real_data = getData()
    print("Data loaded successfully!")

    article_embeddings = word_articles.dot(word_emb)

    emb_times_publication = np.dot(article_embeddings, publication_emb.reshape(100,1))

    article_bias = word_articles.dot(word_bias)

    product_with_bias = emb_times_publication + article_bias

    word_counts = word_articles.sum(axis=1).reshape(word_articles.shape[0], 1)

    final_logits = np.divide(product_with_bias, word_counts) + float(publication_bias)

    indices = final_logits.argsort(axis=0)[-75:].reshape(75)

    word_logits = np.dot(word_emb, publication_emb.reshape(100,1)) + word_bias

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
        for top_word in current_sorted_words[-10:]:
            word = id_to_word[str(top_word)]
            top_words.append(word)
        for least_word in current_sorted_words[:10]:
            word = id_to_word[str(least_word)]
            least_words.append(word)
        current_article['top_words'] = top_words
        current_article['least_words'] = least_words
        return_articles.append(current_article)
        i += 1
    ordered_return_articles = return_articles[::-1]
    response = {
        "statusCode": 200,
        "body": json.dumps(ordered_return_articles)
    }
    return response


if __name__ == "__main__":
    test_event = {
        'a': 5,
        'b': 6,
        'c': 100,
        'd': 12,
        'e': -123
    }
    print(lambda_handler(test_event, ''))
