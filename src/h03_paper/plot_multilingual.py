import os
import sys
import argparse
import pandas as pd

import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import constants
from utils import plot as utils_plot


# LANGUAGES = ['fi', 'ge', 'gr', 'he', 'it', 'ko', 'sp', 'tr']
LANGUAGES = {'fi': 'Finnish', 'ge': 'German', 'gr': 'Greek',
             'he': 'Hebrew', 'it': 'Italian', 'sp': 'Spanish', 'tr': 'Turkish'}
PRETTY_NAMES = {'mgpt': 'mGPT'}
PRED_NAMES = {'surprisal_buggy': 'Surprisal (buggy)', 
              'surprisal': 'Surprisal (corrected)'}

def get_args():
    parser = argparse.ArgumentParser()
    # Results
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    # Model
    parser.add_argument('--dataset', type=str, nargs='+', default='meco')
    parser.add_argument('--languages', type=str, nargs='+', default=list(LANGUAGES.keys()))
    parser.add_argument('--models', type=str, nargs='+', default=['mGPT'])

    return parser.parse_args()


def get_entropy_llh(model, language, dataset, input_path):
    dfs = []
    fname = f'{input_path}/{language}/llh-{dataset}-{model}.tsv'

    try:
        df = pd.read_csv(fname, sep='\t')
    except FileNotFoundError:
        print(fname, ' not found')
        return pd.DataFrame()

    df['language'] = LANGUAGES[language]
    df['model'] = model
    df['model_family'] = PRETTY_NAMES.get(model.split('-')[0], model.split('-')[0])
    df['dataset'] = dataset
    df['dataset'] = df.dataset.apply(lambda x: constants.DATASET_NAMES[x])
    df['name'] = df.name.apply(lambda x: PRED_NAMES.get(x,x))

    drop_keywords = ['prev', 'budget', 'delta', 'next']
    df = df[df.name.apply(lambda x: all([keyword not in x for keyword in drop_keywords]))]
    df = df[(df.predictor_type == 1)]

    return df


def plot_languages(args):
    all_dfs = []
    for language in args.languages:
        for model in args.models:
            df = get_entropy_llh(model, language, args.dataset, args.input_path)
            all_dfs.append(df)
    all_dfs = pd.concat(all_dfs)
    
    all_dfs.sort_values(['language', 'name','model_family'], inplace=True)
    all_dfs['diff'] = all_dfs['diff_empty'] * 100
    
    fig = plt.figure(figsize=(7, 3))
    sns.set_theme(font="DejaVu Serif", style="whitegrid")
    plt.tight_layout()

    g = sns.barplot(all_dfs, x='language', y='diff', hue='name', errorbar=None)
    plt.ylabel(r'$\Delta_{\mathrm{llh}}$ ($10^{-2}$ nats)', fontsize=13)
    plt.xlabel('')

    g.legend()
    for label in g.get_xticklabels():
        label.set_fontsize(10)
    plt.setp( g.xaxis.get_majorticklabels(), rotation=10)
    plt.subplots_adjust(bottom=0.15, left=.15, right=.95, top=.95)
    fig.savefig(f'{args.output_path}/multilingual_llh.pdf', dpi=300)


def main():
    args = get_args()
    utils_plot.config_plots(width=4, height=6)

    plot_languages(args)


if __name__ == '__main__':
    main()