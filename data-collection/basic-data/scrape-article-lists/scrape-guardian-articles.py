#!/usr/bin/env python

OUTPUT_FILE = './guardian-articles-info.json'
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


@dataclass_json
@dataclass
class GuardianArticle:
    title: str = ''
    authors: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    text: str = ''
    url: str = ''
    retrieved: datetime = None


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
        kw_node = soup.find('meta', attrs={'name': 'keywords'})
        kw = kw_node['content'].split(',') if kw_node else []

        ga = GuardianArticle()
        ga.url = url['url']
        ga.title = url['title']
        ga.text = article.text
        ga.keywords = kw
        ga.authors = article.authors
        ga.retrieved = datetime.now()
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
