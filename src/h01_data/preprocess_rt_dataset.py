import os
import sys
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dataset import BrownDataset, NaturalStoriesDataset, ProvoDataset, DundeeDataset, MecoDataset
from utils import utils


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--language', type=str, default='english')
    # Output
    parser.add_argument('--output-fname', type=str, required=True)

    return parser.parse_args()


def get_text(args):
    if args.dataset == 'natural_stories':
        df = NaturalStoriesDataset.preprocess(args.input_path)
    elif args.dataset == 'brown':
        df = BrownDataset.preprocess(args.input_path)
    elif args.dataset == 'provo':
        df = ProvoDataset.preprocess(args.input_path)
    elif args.dataset == 'dundee':
        df = DundeeDataset.preprocess(args.input_path)
    elif args.dataset == 'meco':
        df = MecoDataset.preprocess(args.input_path, args.language)
    else:
        raise ValueError('Invalid dataset name: %s' % args.dataset)

    return df


def main():
    args = get_args()
    df = get_text(args)
    utils.write_tsv(df, args.output_fname)


if __name__ == '__main__':
    main()
