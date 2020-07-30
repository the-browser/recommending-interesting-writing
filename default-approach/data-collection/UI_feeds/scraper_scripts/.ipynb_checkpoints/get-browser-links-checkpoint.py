OUTPUT_FILE = '../scrape_data/browser-rss-urls.json'
WORKER_THREADS = 32

with open('../scrape_data/list_of_links.txt', 'r') as file:
    ARTICLE_LINK_PAGES = file.readlines()

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
from pathlib import Path
import pandas as pd


@dataclass_json
@dataclass
class BrowserRssArticleUrl:
    url: str
    title: str

class WriteThread(Thread):
    def __init__(self, queue: Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue

    def run(self):
        output_file_path = Path(OUTPUT_FILE)
        existing_links = []
            
        while True:
            article = self.queue.get()

            if article is None:
                check_df = pd.DataFrame(existing_links)
                check_df.drop_duplicates(subset="url", keep="first", inplace=True)
                check_df.to_json(output_file_path, orient="records")
                break

            current_article_json = article.to_dict()
            existing_links.insert(0,current_article_json)

class ScrapeThread(Thread):
    def __init__(self, chunk, queue: Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk = chunk
        self.queue = queue

    def run(self):
        for i in self.chunk:
            try:
                print(f'Getting articles from feed {ARTICLE_LINK_PAGES[i]}')
                article_list_page = get(f"{ARTICLE_LINK_PAGES[i]}")
                soup = BeautifulSoup(article_list_page.text, "xml")
                articles = soup.find_all('item')
                for article in articles:
                    link = article.find('link')
                    title = article.find('title')
                    if title is None or title.string is None or link is None or link.string is None:
                        continue
                    article_url = BrowserRssArticleUrl(url=link.string.strip(), title=str(title.string.strip()) or '')
                    self.queue.put(article_url)
            except Exception as e:
                print(f'Something went wrong when scraping: {e}')
                print("------------------------------------------")


if __name__ == '__main__':
    queue = Queue()

    write_thread = WriteThread(queue)
    write_thread.start()

    worker_threads = []
    
    chunk_size =  len(ARTICLE_LINK_PAGES) // WORKER_THREADS
    for i in range(0, len(ARTICLE_LINK_PAGES), chunk_size):
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
