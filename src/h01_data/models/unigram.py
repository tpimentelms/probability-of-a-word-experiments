import csv
from collections import defaultdict
from string import punctuation
import numpy as np
import mosestokenizer


MOSESTOKENIZER = mosestokenizer.MosesTokenizer("en")
MOSESDETOKENIZER = mosestokenizer.MosesDetokenizer("en")


class UnigramModel:
    # pylint: disable=too-few-public-methods

    def __init__(self, lang='en'):
        self.lang = lang
        if lang in ['en', 'english']:
            with open('corpora/rt/unigrams.csv', mode='r', encoding='utf8') as infile:
                reader = csv.reader(infile)
                self.lookup = {rows[0]:-float(rows[1]) for rows in reader}
        else:
            ValueError(f'Unknown language received {lang}')

    def __getitem__(self, key):
        if key == '':
            return np.nan

        try:
            tokens = [MOSESDETOKENIZER([t]) for t in MOSESTOKENIZER(key)]
            return np.sum([self.lookup.get(k, np.nan) for k in tokens])
        except KeyError:
            return np.nan


class ConstantModel:
    # pylint: disable=too-few-public-methods

    def __init__(self, lang='en'):
        self.lang = lang

    def __getitem__(self, key):
        if key == '':
            return np.nan
        return 1


FREQ_MODEL = None
def frequency(word, lang="en"):
    word = word.strip().strip(punctuation).lower()

    # pylint: disable=global-statement
    global FREQ_MODEL
    if not FREQ_MODEL or FREQ_MODEL.lang != lang:
        if lang in ['en', 'english']:
            FREQ_MODEL = UnigramModel(lang)
        else:
            FREQ_MODEL = ConstantModel(lang)
    return FREQ_MODEL[word]
