# recommending-interesting-writing
submission to KDD 2020 IRS workshop

Process:

1. Collecting Data
 
Data was scraped from various online publications, including from publications focused on both feature writing and those who primarily do traditional journalism.
 
2. Training

Training data was sent through training process in batches with an even number of positive and negative examples for each batch. Evaluation recall (percentage correctly labelled of top 150 predictions) were calculated every 50 steps, and model was chosen at the point at which it exhibited maximum recall on both evaluation and test sets.

3. Lambda Deployment

To set up a proper website demo that could fetch articles, we needed a fast approach to generating predictions. We converted all datasets to matrices, including word-embeddings and words in each article, then utilized optimized matrix multiplication to generate new results based on user inputted values for publication embedding. We also fetched top and least contributing words to provide more detail about why articles were rated highly.

4. GitHub Pages

The website was then deployed to github pages and you can access it here: https://the-browser.github.io/recommending-interesting-writing/.
The setup allows you to control the most heavily weighted dimensions of the publication embedding and fetch new results. We also introduced a coronavirus tab that can help you discover more intriguing writing pertaining to the virus.
