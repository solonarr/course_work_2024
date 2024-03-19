"""
Crawler implementation
"""
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Pattern, Union

import requests
from bs4 import BeautifulSoup


class Config:
    """
    Unpacks and validates configurations
    """
    seed_urls: list[str]
    total_articles_to_find_and_parse: int
    headers: dict[str, str]
    encoding: str
    timeout: int
    verify_certificate: bool
    headless_mode: bool

    def __init__(self, path_to_config) -> None:
        """
        Initializes an instance of the Config class
        """
        self.path_to_config = path_to_config
        config_json = self._extract_config_content()
        self._seed_urls = config_json['seed_urls']
        self._num_articles = config_json['total_articles']
        self._headers = config_json['headers']
        self._encoding = config_json['encoding']
        self._timeout = config_json['timeout']
        self._should_verify_certificate = config_json['should_verify_certificate']
        self._headless_mode = config_json['headless_mode']

    def _extract_config_content(self):
        """
        Returns config values
        """
        with open(self.path_to_config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config

    def get_seed_urls(self) -> list[str]:
        """
        Retrieve seed urls
        """
        return self._seed_urls

    def get_num_articles(self) -> int:
        """
        Retrieve total number of articles to scrape
        """
        return self._num_articles

    def get_headers(self) -> dict[str, str]:
        """
        Retrieve headers to use during requesting
        """
        return self._headers

    def get_encoding(self) -> str:
        """
        Retrieve encoding to use during parsing
        """
        return self._encoding

    def get_timeout(self) -> int:
        """
        Retrieve number of seconds to wait for response
        """
        return self._timeout

    def get_verify_certificate(self) -> bool:
        """
        Retrieve whether to verify certificate
        """
        return self._should_verify_certificate

    def get_headless_mode(self) -> bool:
        """
        Retrieve whether to use headless mode
        """
        return self._headless_mode


def make_request(url: str, config: Config) -> requests.models.Response:
    """
    Delivers a response from a request
    with given configuration
    """
    headers = config.get_headers()
    timeout = config.get_timeout()
    verify = config.get_verify_certificate()
    response = requests.get(url, headers=headers, timeout=timeout, verify=verify)
    response.encoding = 'utf-8'
    time.sleep(random.randint(1, 10))
    return response


class Crawler:
    """
    Crawler implementation
    """

    url_pattern: Union[Pattern, str]

    def __init__(self, config: Config) -> None:
        """
        Initializes an instance of the Crawler class
        """
        self.seed_urls = config.get_seed_urls()
        self.config = config
        self.urls = []

    def _extract_url(self, article_bs: BeautifulSoup) -> str:
        """
        Finds and retrieves URL from HTML
        """

        url = article_bs.get('href')
        if isinstance(url, str):
            return url
        return ''

    def find_articles(self) -> None:
        """
        Finds articles
        """
        for link in self.seed_urls:
            response = make_request(link, self.config)
            if response.status_code != 200:
                continue

            main_bs = BeautifulSoup(response.text, 'lxml')
            paragraphs = main_bs.find_all('span',
                                          {'class':
                                               'schema_org'
                                           }
                                          )

            for each_par in paragraphs:
                if len(self.urls) > self.config.get_num_articles():
                    return
                ans = each_par.find_all('a')
                for elem in ans:
                    link = self._extract_url(article_bs=elem)
                    if not link or link in self.urls:
                        continue
                    self.urls.append(link)

    def get_search_urls(self) -> list:
        """
        Returns seed_urls param
        """
        return self.seed_urls


class HTMLParser:
    """
    ArticleParser implementation
    """

    def __init__(self, full_url: str, article_id: int, config: Config) -> None:
        """
        Initializes an instance of the HTMLParser class
        """
        self.full_url = full_url
        self.article_id = article_id
        self.config = config
        self.article = {'art_url': self.full_url, 'art_id': self.article_id, 'art_text': ''}

    def _fill_article_with_text(self, article_soup: BeautifulSoup) -> None:
        """
        Finds text of article
        """
        article_body = article_soup.find_all('div',
                                          {'class':
                                               'article__text'
                                           }
                                          )
        art_text = ' '.join(i.text for i in article_body)
        art_text = re.sub(r'.*Новости\.', '', art_text)
        art_text = re.sub(r'.*Sputnik\.', '', art_text)
        self.article['art_text'] = art_text

    def parse(self):
        """
        Parses each article
        """
        page = make_request(self.full_url, self.config)
        articles = BeautifulSoup(page.content, "lxml")
        self._fill_article_with_text(articles)
        return self.article


def prepare_environment(base_path) -> None:
    """
    Creates ASSETS_PATH folder if no created and removes existing folder
    """
    if Path(base_path).exists():
        shutil.rmtree(Path(base_path))
    Path(base_path).mkdir(parents=True)


class CrawlerRecursive(Crawler):
    """
    An implementation of a recursive crawler
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the instance of CrawlerRecursive class
        """
        super().__init__(config)
        self.start_url = config.get_seed_urls()[0]
        self.counter = 0
        self.visited_urls = []
        self.urls = []
        self.path = Path(__file__).parent / 'ria_crawler_data.json'
        self.load_data()

    def load_data(self) -> None:
        """
        Loads collected data from a file
        """
        if self.path.exists():
            with open(self.path, 'r', encoding=self.config.get_encoding()) as f:
                data = json.load(f)
                self.urls = data['urls']
                self.counter = data['counter']
                self.visited_urls = data['visited_urls']

    def save_data(self) -> None:
        """
        Saves collected data to a file
        """
        data = {'urls': self.urls,
                'counter': self.counter,
                'visited_urls': self.visited_urls}
        with open(self.path, 'w', encoding=self.config.get_encoding()) as f:
            json.dump(data, f, indent=4)

    def find_articles(self) -> None:
        """
        Searching for articles recursively
        """
        if self.start_url not in self.visited_urls:
            self.visited_urls.append(self.start_url)

        response = make_request(self.start_url, self.config)

        main_bs = BeautifulSoup(response.text, 'lxml')

        paragraphs = main_bs.find_all('span',
                                          {'class':
                                               'schema_org'
                                       }
                                      )

        for each_par in paragraphs:
            res = each_par.find_all('a')
            for one in res:
                link = self._extract_url(one)
                if len(self.urls) >= self.config.get_num_articles():
                    return
                if link and link not in self.urls:
                    self.urls.append(link)

        self.save_data()
        self.counter += 1
        self.find_articles()


def main_recursion() -> None:
    """
    Demonstrates the work or recursive crawler
    """
    configuration = Config(path_to_config='RIA_scrapper_config.json')
    prepare_environment('tmp/RIA_vybory_articles')
    r_crawler = CrawlerRecursive(configuration)
    r_crawler.find_articles()
    print(len(r_crawler.urls))
    for i, full_url in enumerate(r_crawler.urls, start=1):
        parser = HTMLParser(full_url=full_url, article_id=i, config=configuration)
        article = parser.parse()
        if isinstance(article, dict):
            article_txt_name = f"{i}_raw.txt"
            with (open(Path(f"tmp/RIA_vybory_articles/{article_txt_name}"), 'w', encoding='utf-8') as file):
                file.write(article['art_text'])


if __name__ == "__main__":
    main_recursion()
