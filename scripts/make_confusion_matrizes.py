import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def make_confusion_matrix(predictions: pd.DataFrame, split: pd.DataFrame) -> np.ndarray:
    predictions = predictions[[
        predictions.columns[0], predictions.columns[-1]]]
    predictions.columns = ['path', 'prediction']
    predictions['path'] = predictions['path'].astype(str)
    predictions['prediction'] = predictions['prediction'].astype(int)
    predictions.set_index('path')

    split.columns = ['path', 'ground_truth']
    split['path'] = split['path'].astype(str)
    split['ground_truth'] = split['ground_truth'].astype(int)
    split.set_index('path')

    assert len(predictions) == len(
        split), f"There are {len(split) - len(predictions)} predictions missing!"

    joined = pd.merge(predictions, split)
    confusion_matrix = metrics.confusion_matrix(
        joined['ground_truth'], joined['prediction'])
    return confusion_matrix


def confusion_matrix_display(confusion_matrix: np.ndarray, labels: List[str], *, save_path: Optional[str] = None) -> None:
    display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=labels)
    if save_path is None:
        display.plot()
        plt.show()
    else:
        display.plot()
        plt.savefig(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--split-file', required=True, nargs='+')
    parser.add_argument('--labels', required=False, default=None, nargs='+')
    parser.add_argument('--output-file', required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(args.input_file, sep=' ', header=None)

    split = [pd.read_csv(path, sep=' ', header=None)
             for path in args.split_file]
    split = pd.concat(split)

    confusion_matrix = make_confusion_matrix(predictions, split=split)

    confusion_matrix_display(
        confusion_matrix=confusion_matrix, labels=args.labels)

    df_cm = pd.DataFrame(confusion_matrix)
    df_cm.to_csv(args.output_file, header=None, index=None, sep=' ')
    print(f'Saved to {args.output_file}')
