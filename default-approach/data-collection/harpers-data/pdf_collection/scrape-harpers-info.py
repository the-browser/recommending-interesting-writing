#!/usr/bin/env python

ARTICLE_URL_FILE = 'harpers-later-urls.json'
OUTPUT_FILE = 'harpers-pdf-links.json'
WORKER_THREADS = 16

import json
import datetime
import dateutil.parser
import sys
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from datetime import datetime
from newspaper import Article, Config
from bs4 import BeautifulSoup
from typing import List
from queue import Queue
from threading import Thread
from pathlib import Path
import pandas as pd
import re
import time
from urllib.request import Request, urlopen

@dataclass_json
@dataclass
class HarpersPDF:
    title: str = ''
    url: str = ''
    link: str = ''
    publication: str = 'harpers'
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

        i = 0
        while True:
            current_pdf = self.queue.get()
            if current_pdf is None:
                check_df = pd.DataFrame(existing_info)
                check_df.to_json(output_file_path, orient="records")
                print("Saved File!")
                break

            current_pdf_json = current_pdf.to_dict()
            stringed_path = str(i) + "-document.pdf"
            current_pdf_path = Path('raw_pdfs') / stringed_path
            req = Request(current_pdf_json['url'] , headers={'User-Agent': 'Mozilla/5.0'})
            with open(current_pdf_path, 'wb') as file:
                file.write(urlopen(req).read())
            existing_info.insert(0,current_pdf)
            i += 1
            
class ScrapeThread(Thread):
    def __init__(self, urls, queue: Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.urls = urls
        self.queue = queue

    def run(self):
        for url in self.urls:
            try:
                print(f"scraping {url['url']}")
                req = Request(url['url'] , headers={'User-Agent': 'Mozilla/5.0'})
                webpage = urlopen(req).read()
                soup = BeautifulSoup(webpage, "html5lib")
                if soup.find_all("div", {"class": "pdf-only"}):
                    new_pdf = HarpersPDF()
                    new_pdf.title = url['title']
                    new_pdf.url = soup.find('a', {'class': 'btn-primary'})['href']
                    new_pdf.link = new_pdf.url
                    self.queue.put(new_pdf)
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
