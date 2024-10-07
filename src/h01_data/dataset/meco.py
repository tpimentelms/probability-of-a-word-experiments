import numpy as np
import pandas as pd

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri    
from rpy2.robjects.packages import importr

from .base import BaseDataset
from utils import utils


class MecoDataset(BaseDataset):
    languages = ['fi', 'ge', 'gr', 'he', 'it', 'ko', 'en',
                 'sp', 'tr']
    # 'ru': df[df.lang.isna()] reveals a bug in the Russian data.
    # 'en': (df.drop_duplicates(['text_id', 'word_id', 'word']).shape[0] == 
    #        df.drop_duplicates(['text_id', 'word_id']).shape[0]) reveals a bug in English data.
    # 'ee': not in mGPT
    # 'no': not in mGPT
    # 'du': (maybe) not in mGPT? https://huggingface.co/ai-forever/mGPT says no, plot says yes.
    kept_columns_initial = {'trialid': 'text_id', 'ianum': 'word_id', 'ia': 'word', 
                            'skip': 'skipped', 'firstrun.dur': 'time', 
                            'uniform_id': 'worker_id'}
    unused_columns_final = ['word', 'WorkerId']

    @classmethod
    def get_text(cls, input_path, language):
        stories = cls.get_stories_text(input_path, language)
        stories_text = [x[1].strip() for x in stories]

        return '\n'.join(stories_text)
    
    @classmethod
    def preprocess(cls, input_path, language):
        # Get natural stories text
        stories = cls.get_stories_text(input_path, language)
        ns_text_words = cls.get_corpus_words(stories)

        # Get RTs data
        df = cls.read_data(input_path, language)
        df['word_id'] = df['word_id'] - 1

        # Exclude outliers
        df = utils.find_outliers(df, transform=np.log)

        # Fix bug in the numbering of word_id's in Spanish
        if language == 'sp':
            df.loc[(df.text_id == 9) & (df.word_id > 124), 'word_id'] = \
                df.loc[(df.text_id == 9) & (df.word_id > 124), 'word_id'] - 1 
            assert (df.drop_duplicates(['text_id', 'word_id', 'word']).shape[0] == 
                    df.drop_duplicates(['text_id', 'word_id']).shape[0])

        # Create preprocessed dataframe
        df = cls.create_analysis_dataframe(df, ns_text_words)

        # Check word matches ref_token
        assert (df['word'] == df['ref_token']).all()

        # Deleted unused info from dataframe
        cls.remove_unused_columns(df, cls.unused_columns_final)
        return df

    @classmethod
    def get_stories_text(cls, input_path, language):
        df = cls.read_data(input_path, language)

        # Drop duplicates and check no two words have same id. Then sort based on ids.
        df.drop_duplicates(['text_id', 'word_id', 'word'], inplace=True)
        assert (df.shape[0] == df.drop_duplicates(['text_id', 'word_id']).shape[0])
        df.sort_values(['text_id', 'word_id'], inplace=True)

        # Concatenate stories
        df_stories = df.groupby('text_id')['word'].aggregate(lambda x: ' '.join(x))
        paragraphs = [(text_id, text_str) for text_id, text_str in df_stories.items()]

        return paragraphs

    @classmethod
    def read_data(cls, input_path, language):
        # Assert language is in dataset
        assert language in cls.languages, f'Language {language} not in dataset.'

        # Read data
        df = cls.read_rda_file(input_path)

        # Filter based on language, drop unused columns and rename
        df = df[df.lang == language]
        df = df[cls.kept_columns_initial.keys()]
        df.rename(columns={'trialid': 'text_id', 'ianum': 'word_id', 'ia': 'word', 
                           'skip': 'skipped', 'firstrun.dur': 'time', 
                           'uniform_id': 'WorkerId'}, inplace = True)
        df = df.astype({'text_id': 'Int64', 'word_id': 'Int64'})
        
        # Set skipped words reading times to 0
        assert (df.time.isna() == df.skipped).all()
        df.loc[df.skipped == 1, 'time'] = 0

        return df.copy()
    
    @staticmethod
    def read_rda_file(input_path):
        pandas2ri.activate()

        base = importr('base')
        # base.load(f'{input_path}/joint_fix_trimmed.rda')
        base.load(f'{input_path}/joint_data_trimmed.rda')
        rdf_List = base.mget(base.ls())

        # ITERATE THROUGH LIST OF R DFs 
        pydf_dict = {}
        for i,f in enumerate(base.names(rdf_List)):
            pydf_dict[f] = pandas2ri.rpy2py_dataframe(rdf_List[i])

        return pydf_dict['joint.data']
