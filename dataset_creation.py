from transformers import MBartTokenizer, MBartForConditionalGeneration

import json

from rake_nltk import Rake
from nltk.corpus import stopwords
from pymystem3 import Mystem
import spacy
from string import punctuation
from operator import floordiv
from itertools import islice

from pathlib import Path

# import random

mbart_ru_sum_gazeta_tokenizer = MBartTokenizer.from_pretrained("D:\\mbart_ru_sum_gazeta")
mbart_ru_sum_gazeta_model = MBartForConditionalGeneration.from_pretrained("D:\\mbart_ru_sum_gazeta")


class KeySntExtractor:

    def __init__(self, text):
        self.text = text

    def extract_with_rake(self, rake: Rake()):
        rake.extract_keywords_from_text(self.text)
        return rake.get_ranked_phrases_with_scores()

    @staticmethod
    def prepare_string(string):
        string = string.replace(" ", "")
        string = string.lower()
        return string

    @staticmethod
    def take(n, iterable):
        """Return the first n items of the iterable as a list."""
        return list(islice(iterable, n))

    def create_key_text(self):
        strange_text = []
        result_dict = {}
        new_text = self.text.split('. ')
        keywords = self.extract_with_rake(rake=Rake())

        for sn in new_text:
            new_sn = self.prepare_string(sn)
            strange_text.append(new_sn)

        for snt in keywords:
            result_dict[snt[1]] = [key_phrase for key_phrase, strange_phrase in zip(new_text, strange_text) if
                                   self.prepare_string(snt[1]) in strange_phrase]

        n = floordiv(len(new_text), 3)

        resulted_snts = self.take(n, result_dict.items())

        resulted_text = [''.join(phrase[1]) for phrase in resulted_snts]

        return '. '.join(resulted_text)


# Create lemmatizer and stopwords list
mystem = Mystem()
russian_stopwords = stopwords.words("russian")


# Preprocess function
def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords
              and token != " "
              and token.strip() not in punctuation]

    text = " ".join(tokens)

    return text


path_to_articles = Path('tmp/TASS_sport_chess_articles')
doc_list_2 = []
resulted_text_4 = []

train_text = ''
for child in list(islice(path_to_articles.iterdir(), 4)):
    with child.open('r', encoding='utf-8') as f:
        doc_list_2.append(f.readline())


# creating key_texts from doc list then merge them into one text

for text in doc_list_2:
    key_snt_extractor = KeySntExtractor(text)
    key_text = key_snt_extractor.create_key_text()
    resulted_text_4.append(key_text)

resulted_splitted_4 = []

for text in resulted_text_4:
    new_text = text.split('. ')
    # random.shuffle(new_text)
    for snt in new_text:
        resulted_splitted_4.append(snt.strip(' .'))


resulted_splitted_4 = list(set(resulted_splitted_4))

resulted_splitted_4_main_copy = resulted_splitted_4.copy()
resulted_splitted_4_copy = resulted_splitted_4.copy()
new_resulted_splitted_4 = resulted_splitted_4.copy()

nlp = spacy.load('ru_core_news_lg')


for snt in resulted_splitted_4_main_copy:
    preprocessed_snt = preprocess_text(snt)
    snt1 = nlp(preprocessed_snt)
    for string in resulted_splitted_4_copy:
        preprocessed_string = preprocess_text(string)
        snt2 = nlp(preprocessed_string)
        simi = snt2.similarity(snt1)
        if 0.96 <= simi < 1.0:
            if len(snt) >= len(string):
                if string in new_resulted_splitted_4:
                    new_resulted_splitted_4.remove(string)
            else:
                if snt in new_resulted_splitted_4:
                    new_resulted_splitted_4.remove(snt)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield '. '.join(lst[i:i + n])


splited_in_parts = (list(chunks(resulted_splitted_4, 6)))

if len(''.join(resulted_splitted_4)) > 2048:
    print('Too long!')
else:
    resulted_text_5_copy = ''

    # summarize each doc then merge summaries into one text

    for text in splited_in_parts:
        max_length = 0
        for snt in text.split('. '):
            max_length += len(snt.split())

        max_length *= 2

        min_length = int(max_length / 3)

        tokens = mbart_ru_sum_gazeta_tokenizer(
            [text],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
        generation = mbart_ru_sum_gazeta_model.generate(
            input_ids=tokens,
            min_length=min_length,
            max_length=max_length,
            no_repeat_ngram_size=4
        )[0]
        resulted_text_5_copy += f"{mbart_ru_sum_gazeta_tokenizer.decode(generation, skip_special_tokens=True)}"

    print(f"{splited_in_parts}:\n{resulted_text_5_copy}")

    dict = {"text": ''.join(splited_in_parts),
            "summary": resulted_text_5_copy}

    with open('news_data.json', 'a', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)
