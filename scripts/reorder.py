import argparse
import numpy as np
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--x-order', required=True, nargs='+')
    parser.add_argument('--y-order', required=True, nargs='+')
    parser.add_argument('--output-file', required=True)
    args = parser.parse_args()

    assert len(args.x_order) == len(args.y_order)

    cm = pd.read_csv(args.input_file, sep=' ',
                     header=None, index_col=None).to_numpy()

    x_order = np.array(list(map(int, args.x_order)))
    y_order = np.array(list(map(int, args.y_order)))

    print(cm)

    cm = cm[x_order, :]
    cm = cm[:, y_order]

    print(cm)

    print(x_order)
    print(y_order)
