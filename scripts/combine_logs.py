import argparse
from enum import Enum
from functools import reduce
import glob
import os
import re
from typing import Dict, List, Tuple, Union

import pandas as pd

from patterns import split_paths, ranked_confidences_pattern, ranked_confusion_matrix_pattern, ranked_votes_pattern


class CombinationType(Enum):
    ConfusionMatrix = 'confusion_matrix'
    Confidences = 'confidences'
    Votes = 'votes'


def combine_confusion_matrizes(path_infos: List[Dict[str, str]]) -> List[Dict[str, Union[int, pd.DataFrame]]]:
    for path_info in path_infos:
        assert 'path' in path_info and 'rank' in path_info and 'split' in path_info

    combined = {}
    for path_info in path_infos:
        split = path_info['split']
        if split not in combined:
            combined[split] = []
        combined[split].append(path_info['path'])
    return [(split, sum([pd.read_csv(path, header=None, index_col=False, sep=' ') for path in paths])) for split, paths in combined.items()]


def combine_confidences_and_predict(confidence_csv_paths) -> List[Tuple[int, pd.DataFrame]]:

    def _combine_confidences_and_predict(paths) -> pd.DataFrame:
        dfs = [pd.read_csv(path, sep=' ', header=None) for path in paths]
        df = pd.concat(dfs)
        df.columns = ['path', *df.columns[1:-1], 'prediction']
        df = df.groupby(['path']).agg(
            {column: 'sum' for column in df.columns[1:-1]}).reset_index()
        df['prediction'] = df[df.columns[1:]].idxmax(axis=1) - 1
        return df

    paths = [(path, int(ranked_confidences_pattern.match(path).group('split')))
             for path in confidence_csv_paths]
    splits = set(map(lambda x: x[1], paths))
    paths = [(x, [y[0]for y in paths if y[1] == x]) for x in splits]

    return [(split, _combine_confidences_and_predict(path)) for (split, path) in paths]


def combine_votes_and_predict(vote_paths) -> List[Tuple[int, pd.DataFrame]]:

    def _combine_votes_and_predict(paths) -> pd.DataFrame:
        dfs = [pd.read_csv(path, sep=' ', header=None) for path in paths]
        df = pd.concat(dfs)
        df.columns = ['path', *df.columns[1:-1], 'prediction']
        df = df.groupby(['path']).agg(
            {column: 'sum' for column in df.columns[1:-1]}).reset_index()
        df['prediction'] = df[df.columns[1:]].idxmax(axis=1) - 1
        return df

    paths = [(path, int(ranked_votes_pattern.match(path).group('split')))
             for path in vote_paths]
    splits = set(map(lambda x: x[1], paths))
    paths = [(x, [y[0]for y in paths if y[1] == x]) for x in splits]

    return [(split, _combine_votes_and_predict(path)) for (split, path) in paths]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument(
        '--type', choices=[e.value for e in CombinationType], required=True)
    parser.add_argument('--glob-pattern', required=True)
    parser.add_argument('--info-pattern', required=True)
    args = parser.parse_args()

    combination_type = CombinationType(args.type)
    file_pattern = os.path.join(args.path, args.info_pattern)
    file_pattern = re.compile(file_pattern)

    glob_pattern = os.path.join(args.path, args.glob_pattern)
    paths = glob.glob(glob_pattern)

    path_infos = [dict(path=path, **file_pattern.match(path).groupdict())
                  for path in paths]

    type = CombinationType(args.type)
    if type == CombinationType.ConfusionMatrix:

        combined_confusion_matrix_infos = combine_confusion_matrizes(
            path_infos=path_infos)

        for split, cm in combined_confusion_matrix_infos:

            output_path = os.path.join(
                args.path, f'clip_{split}_confusion_matrix.csv')
            cm.to_csv(output_path, header=False, index=False, sep=' ')
            print(f'Wrote confusion matrix to {output_path}')

    # # Be careful as these combined confusion matrizes don't actually have any meaning (as we don't know which video was predicted when and where)

    # cms = combine_confusion_matrizes(confusion_matrix_paths)
    # for split, cm in cms:
    #     pd.DataFrame(cm).to_csv(os.path.join(
    #         args.path, f'{split}_confusion_matrix.csv'), header=False, index=False)

    # combined_confidences = combine_confidences_and_predict(
    #     confidence_csv_paths)

    # confidence_based_votes = combine_confidences_and_predict(
    #     confidence_csv_paths)

    # for split, votes in confidence_based_votes:
    #     path = os.path.join(args.path, f'{split}_confidences.csv')
    #     votes.to_csv(path, header=False, index=False, sep=' ')
    #     print(f'Saving confidences file to: {path}')

    # direct_votes = combine_votes_and_predict(vote_csv_paths)

    # for split, votes in direct_votes:
    #     path = os.path.join(args.path, f'{split}_votes.csv')
    #     votes.to_csv(path, header=False, index=False, sep=' ')
    #     print(f'Saving votes file to: {path}')
