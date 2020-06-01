#!/usr/bin/env python

BASE_URL = "https://www.vox.com/fetch/archives/"
YEARS = [str(month) for month in range(2014,2020)]
MONTHS = [str(month) for month in range(1,13)]
OUTPUT_FILE = './vox-article-urls.json'
WORKER_THREADS = 16

import json
import datetime
import dateutil.parser
from itertools import product
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from datetime import datetime
from newspaper import Article
from bs4 import BeautifulSoup
from typing import List
from queue import Queue
from threading import Thread
from requests import get
from time import sleep


@dataclass_json
@dataclass
class VoxArticleUrl:
    url: str
    title: str
    month: int
    year: int

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

    def get_urls(self, year, month):
        page = 1
        while True:
            sleep(1)  # Prevent rate limiting
            print(f'Getting articles for year {year} month {month}, page {page}')
            return_data = get(f"{BASE_URL}/{year}/{month}/{page}", headers={'Accept': 'application/json'})
            if return_data.status_code != 200:
                print(f"Received status {return_data.status_code}")
                if return_data.status_code != 429:
                    return
                else:
                    sleep(10)
            else:
                page = page + 1
                response = return_data.json()
                soup = BeautifulSoup(response['html'], "html5lib")
                yield soup
                if not response['has_more']:
                    break



    def run(self):
        for year, month in self.chunk:
            try:
                for html in self.get_urls(year, month):
                    h2s = html.find_all('h2')
                    for h2 in h2s:
                        a = h2.find('a')
                        title = a.string
                        url = a['href']
                        print(title,url)
                        vox_url = VoxArticleUrl(title=str(title), url=str(url), month=int(month), year=int(year))
                        self.queue.put(vox_url)
            except Exception as e: # Best effort
                print(f'Something went wrong when scraping: {e}')



if __name__ == '__main__':
    queue = Queue()

    write_thread = WriteThread(queue)
    write_thread.start()

    m_y_combos = list(product(YEARS, MONTHS))
    worker_threads = []
    chunk_size = len(m_y_combos) // WORKER_THREADS
    for i in range(0, len(m_y_combos), chunk_size):
        chunk = m_y_combos[i:i+chunk_size]
        worker_threads.append(ScrapeThread(chunk, queue))

    for thread in worker_threads:
        thread.start()

    for thread in worker_threads:
        thread.join()

    # Signal end of jobs to write thread
    queue.put(None)

    print('Done.')
    write_thread.join()


