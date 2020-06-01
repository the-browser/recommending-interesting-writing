#!/usr/bin/env python

ARTICLE_URL_FILE = '../scrape_data/browser-rss-urls.json'
OUTPUT_FILE = '../scrape_data/browser-rss-articles-info.json'
WORKER_THREADS = 16

import json
import datetime
import dateutil.parser
import sys
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from datetime import datetime
from newspaper import Article
from bs4 import BeautifulSoup
from typing import List
from queue import Queue
from threading import Thread
from pathlib import Path
import pandas as pd
import re

@dataclass_json
@dataclass
class LongformArticle:
    title: str = ''
    text: str = ''
    url: str = ''
    link: str = ''
    publication: str = 'browser'
    model_publication: str = 'target'


class WriteThread(Thread):
    def __init__(self, queue: Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue

    def run(self):
        output_file_path = Path(OUTPUT_FILE)
        if output_file_path.is_file():
            with open(OUTPUT_FILE, 'r') as output_file:
                existing_info = json.loads(output_file.read())

        else:
            existing_info = []

        while True:
            article = self.queue.get()
            if article is None:
                check_df = pd.DataFrame(existing_info)
                check_df.drop_duplicates(subset="url", keep="first", inplace=True)
                check_df.to_json(output_file_path, orient="records")
                print("Saved File!")
                break

            current_article_json = article.to_dict()
            existing_info.insert(0,current_article_json)


class ScrapeThread(Thread):
    def __init__(self, urls, queue: Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.urls = urls
        self.queue = queue

    @staticmethod
    def scrape(url):
        article = Article(url['url'])
        article.download()
        article.parse()

        soup = BeautifulSoup(article.html, 'lxml')

        ga = LongformArticle()
        ga.url = url['url']
        ga.title = url['title']
        ga.link = url['url']
        ga.text = article.text
        return ga

    def run(self):
        for url in self.urls:
            try:
                print(f"scraping {url['url']}")
                article = ScrapeThread.scrape(url)
                self.queue.put(article)
            except Exception as e: # Best effort
                print(f'ScrapeThread Exception: {e}')


if __name__ == '__main__':
    urls = []
    try:
        with open(ARTICLE_URL_FILE, 'r') as f:
            urls = json.load(f)
    except:
        print('Error opening article url file')
        sys.exit(1)

    queue = Queue()

    write_thread = WriteThread(queue)
    write_thread.start()

    worker_threads = []
    chunk_size = len(urls) // WORKER_THREADS
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i+chunk_size]
        worker_threads.append(ScrapeThread(chunk, queue))

    for thread in worker_threads:
        thread.start()

    for thread in worker_threads:
        thread.join()

    # Signal end of jobs to write thread
    queue.put(None)

    print('Done.')
    write_thread.join()
