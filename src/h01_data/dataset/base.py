from collections import defaultdict
from string import punctuation
import pandas as pd

from utils import utils


class BaseDataset():
    @staticmethod
    def get_corpus_words(stories):
        stats = defaultdict(dict)
        for i, s in stories:
            # remove leading and trailing white space
            s = s.strip()
            stats['split_string'][i] = s.split()

        return stats

    @classmethod
    def create_analysis_dataframe(cls, df, stats, lang="en", dataset=None):
        df = df.copy()
        # get standard corpus statistics
        df = cls.add_standard_columns(df, stats['split_string'], lang=lang, dataset=dataset)

        return df

    @staticmethod
    def add_standard_columns(df, split_strings, lang="en", dataset=None):
        # ref_token is used for sanity checking. should be same as word
        df['ref_token'] = df[['text_id', 'word_id']].apply(
            lambda x: split_strings[x['text_id']][x['word_id']], axis=1)

        # Center times per worker
        df['centered_time'] = df['time'] - df.groupby(by=["WorkerId"])["time"].transform('mean')

        # Get word length
        df['word_len'] = df['word'].apply(len)

        return df
    
    @classmethod
    def remove_unused_columns(cls, df, unused_columns):
        for col in unused_columns:
            del df[col]
