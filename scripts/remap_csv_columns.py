import argparse
from enum import Enum
import glob
import os
import re
from typing import List, Union

import numpy as np
import pandas as pd
import patterns


class Datasets(Enum):
    RAVDESS = 'RAVDESS'
    CREMA_D = 'CREMA_D'

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


mappings = {
    Datasets.RAVDESS: {
        Datasets.CREMA_D: [4, None, 2, 0, 5, 1, 3, None]
    },
    Datasets.CREMA_D: {
        Datasets.RAVDESS: [3, 5, 2, 6, 0, 4, None, None]
    }
}


def remap(paths: List[str], training_dataset: Datasets, validation_dataset: Datasets, *, repredict: bool = False) -> pd.DataFrame:

    def _remap(path: str, mapping: List[Union[int, None]]) -> pd.DataFrame:

        df = pd.read_csv(path, header=None, sep=' ')
        df.columns = ['path', *(np.array(df.columns[1:-1])) - 1, 'prediction']

        if repredict:
            # Deleting all columns, that do not exist.
            mapping = [idx for idx in mapping if idx is not None]

            # Repredicting with the remaining votes.
            df = df[['path', *mapping]]
            df.columns = ['path', *list(range(len(df.columns) - 1))]
            df['prediction'] = df[df.columns[1:]].idxmax(axis=1)
        else:
            # Mapping all values that are not within the dataset to num_classes + 1, i.e. a collection class.
            nun_counts = mapping.count(None)
            start_idx = max(list(filter(lambda x: x is not None, mapping))) + 1
            end_idx = start_idx + nun_counts
            insertion_idxs = list(range(start_idx, end_idx))
            mapping = [idx if idx is not None else insertion_idxs.pop(
                0) for idx in mapping]
            df['prediction'] = df['prediction'].apply(lambda row: mapping[row])

        return df

    mapping = mappings[training_dataset][validation_dataset]

    return [_remap(path, mapping) for path in paths]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-dataset',
                        choices=Datasets.list(), required=True)
    parser.add_argument('--validation-dataset',
                        choices=Datasets.list(), required=True)
    parser.add_argument('--repredict', action='store_true', default=False)
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()

    if args.training_dataset == args.validation_dataset:
        print(f'No remapping required!')
        quit()

    training_dataset = Datasets(args.training_dataset)
    validation_dataset = Datasets(args.validation_dataset)

    csv_paths = glob.glob(os.path.join(args.dir, f'*.csv'))

    confidences_pattern = re.compile(os.path.join(
        args.dir, patterns.confidences_pattern))
    votes_pattern = re.compile(os.path.join(args.dir, patterns.votes_pattern))

    confidences_paths = [
        (
            int(confidences_pattern.match(path).group('split')),
            path
        )
        for path in csv_paths if confidences_pattern.match(path)
    ]

    votes_paths = [
        (
            int(votes_pattern.match(path).group('split')),
            path
        )
        for path in csv_paths if votes_pattern.match(path)
    ]

    remapped_confidences = remap(
        map(lambda x: x[1], confidences_paths), training_dataset=training_dataset, validation_dataset=validation_dataset, repredict=args.repredict)

    for split, remapped_confidence in zip(map(lambda x: x[0], confidences_paths), remapped_confidences):
        path = os.path.join(args.dir, f'{split}_remapped_confidences.csv')
        remapped_confidence.to_csv(path, header=None, index=None, sep=' ')
        print(f'Saved remapped confidences to: {path}')

    remapped_votes = remap(
        map(lambda x: x[1], votes_paths), training_dataset=training_dataset, validation_dataset=validation_dataset, repredict=args.repredict)

    for split, remapped_vote in zip(map(lambda x: x[0], votes_paths), remapped_votes):
        path = os.path.join(args.dir, f'{split}_remapped_votes.csv')
        remapped_vote.to_csv(path, header=None, index=None, sep=' ')
        print(f'Saved remapped votes to: {path}')
