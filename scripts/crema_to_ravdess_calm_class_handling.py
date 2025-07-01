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
    cm = df.to_numpy()
    if args.combine:
        cm[0, :] += cm[1, :]
    cm = cm[[0, *range(2, 7)], :]
    cm = cm[:, [0, *range(2, 7)]]

    confusion_matrix_display(
        confusion_matrix=cm, labels=args.labels, save_path=args.file[:-4] + ('_combined' if args.combine else '') + '.png')
    metrics = calculate_metrics(cm=cm)
    metrics_save_path = args.file[:-4] + \
        ('_combined_' if args.combine else '_') + 'metrics.yaml'
    with open(metrics_save_path, 'w') as f:
        print(f'Saving to {metrics_save_path}')
        yaml.dump(metrics, f, default_flow_style=False)
