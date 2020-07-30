BASE_URL="https://aeon.co/essays/popular?page="
N_ARTICLE_LINK_PAGES = 50
OUTPUT_FILE = '../article-lists/aeon-article-urls.json'
WORKER_THREADS = 16

import json
import datetime
import dateutil.parser
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from datetime import datetime
from newspaper import Article
from bs4 import BeautifulSoup
from typing import List
from queue import Queue
from threading import Thread
from requests import get


@dataclass_json
@dataclass
class AeonArticleUrl:
    url: str
    title: str

class WriteThread(Thread):
    def __init__(self, queue: Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue

    def run(self):
        with open(OUTPUT_FILE, 'a') as output_file:
            output_file.write("[\n")
            first_entry = True

            while True:
                article = self.queue.get()

                if article is None:
                    output_file.write("\n]")
                    break

                article_json = article.to_json(indent=4)

                if first_entry:
                    first_entry = False
                else:
                    output_file.write(",\n")

                output_file.write(article_json)


class ScrapeThread(Thread):
    def __init__(self, chunk, queue: Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk = chunk
        self.queue = queue

    def run(self):
        for i in self.chunk:
            try:
                print(f'Getting articles from list page {i}')
                article_list_page = get(f"{BASE_URL}{i}")
                soup = BeautifulSoup(article_list_page.text, "html5lib")
                articles = soup.find_all('article', {'class': 'article-card'})
                for article in articles: 
                    link = article.find('a', {'class': 'article-card__link'})
                    title = article.find('a', {'class': 'article-card__title'})
                    print(title.string.strip())
                    if title is None or title.string is None:
                        continue
                    article_url = AeonArticleUrl(url="https://aeon.co" + link['href'], title=str(title.string.strip()) or '')
                    self.queue.put(article_url)
            except Exception as e:
                print(f'Something went wrong when scraping: {e}')
                print("------------------------------------------")


if __name__ == '__main__':
    queue = Queue()

    write_thread = WriteThread(queue)
    write_thread.start()

    worker_threads = []
    chunk_size =  N_ARTICLE_LINK_PAGES // WORKER_THREADS
    for i in range(0, N_ARTICLE_LINK_PAGES+1, chunk_size):
        chunk = range(i,i+chunk_size)
        worker_threads.append(ScrapeThread(chunk, queue))

    for thread in worker_threads:
        thread.start()

    for thread in worker_threads:
        thread.join()

    # Signal end of jobs to write thread
    queue.put(None)

    print('Done.')
    write_thread.join()