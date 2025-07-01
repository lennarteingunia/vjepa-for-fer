import argparse

import pandas as pd
import yaml

from make_confusion_matrizes import confusion_matrix_display
from metrics import calculate_metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--labels', required=False, default=None, nargs='+')
    parser.add_argument('--combine', action='store_true', default=False)
    args = parser.parse_args()

    df = pd.read_csv(args.file, header=None, index_col=False, sep=' ')
    df = df.head(6)
    df.columns = list(range(len(df.columns)))

    if args.combine:
        df[4] += df[6]

    df = df[list(range(6))]
    confusion_matrix = df.to_numpy()
    confusion_matrix_display(
        confusion_matrix=confusion_matrix, labels=args.labels, save_path=args.file[:-4] + ('_combined' if args.combine else '') + '.png')

    metrics = calculate_metrics(cm=confusion_matrix)
    metrics_save_path = args.file[:-4] + \
        ('_combined_' if args.combine else '_') + 'metrics.yaml'
    with open(metrics_save_path, 'w') as f:
        print(f'Saving to {metrics_save_path}')
        yaml.dump(metrics, f, default_flow_style=False)
