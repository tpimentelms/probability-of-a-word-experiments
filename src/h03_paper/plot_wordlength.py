import os
import sys
import argparse
import pandas as pd
import scipy

import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import constants
from utils import plot as utils_plot


MODEL_SIZES = {
        'pythia-70m': 70,
        'pythia-160m': 160,
        'pythia-410m': 410,
        'pythia-14b': 1400,
        'pythia-28b': 2800,
        'pythia-69b': 6900,
        # 'pythia-120b': 12000, ## not enough time to run on same words as other models
    }

PRETTY_NAMES = {'pythia': 'Pythia', 'gpt': 'GPT-2'}
PRED_NAMES = {'surprisal_buggy': 'Surprisal (buggy)', 
              'surprisal': 'Surprisal (corrected)'}
HYP_NAMES = {'piantadosi': r'CCH (Piantadosi et al.)', 
             'cch': r'CCH (Pimentel et al.)',
             'zipf': r'Zipf'}

def get_args():
    parser = argparse.ArgumentParser()
    # Results
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    # Model
    parser.add_argument('--models', type=str, nargs='+', default=list(MODEL_SIZES.keys()))

    return parser.parse_args()


def get_wordlength_correlations(input_path, model):
    dfs = []
    fname_base = os.path.join(input_path, 'lengths-%s.tsv')
    fname = fname_base % (model)

    try:
        df_raw = pd.read_csv(fname, sep='\t')
    except FileNotFoundError:
        print(fname, ' not found')
        return pd.DataFrame()

    del df_raw['Unnamed: 0']
    df_raw['zipf'] = - df_raw['zipf']

    predictors = ['piantadosi_buggy', 'piantadosi', 'cch_buggy', 'cch', 'zipf']
    rows = []
    for predictor in predictors:
        pred_type = 'surprisal_buggy' if 'buggy' in predictor else 'surprisal'
        hypothesis = predictor.replace('_buggy', '')

        spearman, pvalue = scipy.stats.spearmanr(df_raw['length'], df_raw[predictor])
        rows += [[predictor, HYP_NAMES[hypothesis], PRED_NAMES[pred_type], 'Spearman', spearman, pvalue]]
        pearson, pvalue = scipy.stats.pearsonr(df_raw['length'], df_raw[predictor])
        rows += [[predictor, HYP_NAMES[hypothesis], PRED_NAMES[pred_type], 'Pearson', pearson, pvalue]]

    df = pd.DataFrame(rows, columns=['predictor', 'hypothesis', 'name', 'metric', 'value', 'pvalue'])

    df['model'] = model
    df['model_family'] = PRETTY_NAMES.get(model.split('-')[0], model.split('-')[0])
    df['size'] = df.model.apply(lambda x: MODEL_SIZES[x])

    return df


def plot_wordlength_results(args):
    all_dfs = []

    for model in args.models:
        df = get_wordlength_correlations(args.input_path, model)
        all_dfs.append(df)
    all_dfs = pd.concat(all_dfs)

    # zipf_corr = all_dfs.loc[all_dfs.hypothesis == 'Zipf', 'value'].unique()[0]
    # all_dfs.loc[all_dfs.hypothesis == 'Zipf', 'value'] = zipf_corr
    all_dfs.loc[all_dfs.hypothesis == 'Zipf', 'name'] = 'Surprisal (unigram)'
    
    all_dfs.sort_values(['size', 'name', 'predictor', 'model_family'], inplace=True)
    
    # sns.set_context("notebook", font_scale=0.1)

    sns.set_theme(font="DejaVu Serif", style="whitegrid")
    plt.tight_layout()
    plt.xscale('log')

    g = sns.FacetGrid(all_dfs, col="metric", height=2.8, aspect=.65 * (3.5/2.8), sharey=True, gridspec_kws={"wspace":0.3}) #
    g.set_titles(col_template="{col_name}",fontsize=10)

    g.map_dataframe(sns.lineplot, x='size', y='value', hue='name', style='hypothesis', errorbar=('ci', 95), n_boot=20000)
    
    # g.add_legend()
    g.add_legend(bbox_to_anchor=(0.11, 0.08), loc='upper left', borderaxespad=0, ncol=2, fontsize=8)
    g.set_axis_labels("", r'Correlations', fontsize=10)
    g.legend.get_texts()[0].set_text('')
    g.legend.get_texts()[4].set_text('')
    # g.set_xlabel("# of Parameters (in Millions)")
    for i,ax in enumerate(g.axes.flat):

        if i == 0:
            ax.set_xlabel("                                 # of Parameters (in Millions)")
        ax.xaxis.set_tick_params(pad=0)
        ax.yaxis.set_tick_params(pad=0)
        ax.set(xscale="log")
        for label in ax.get_yticklabels():
            label.set_fontsize(9)
        for label in ax.get_xticklabels():
            label.set_fontsize(9)
        
    plt.subplots_adjust(bottom=0.2)
    # plt.legend(bbox_to_anchor=(1.02, 0.15), loc='upper left', borderaxespad=0)
    g.savefig(f'{args.output_path}/wordlength_spearman.pdf', dpi=300)


def main():
    args = get_args()
    utils_plot.config_plots(width=4, height=6)

    plot_wordlength_results(args)


if __name__ == '__main__':
    main()