import glob
from math import floor
import os
import random
from typing import List, Literal
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from jinja2 import Environment, FileSystemLoader
import scipy
import scipy.stats


@click.group('cli')
def cli() -> None:
    pass


@cli.command()
@click.argument('root', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=True))
def analyze(root: str, output_dir: str) -> None:

    click.echo(f'Searching for split files in {output_dir}.')
    pattern = os.path.join(output_dir, '*.csv')
    paths = list(sorted(glob.glob(pattern)))
    click.echo(f'Found a total of {len(paths)} split files in {output_dir}.')

    video_demo_path = os.path.join(root, 'VideoDemographics.csv')
    click.echo(f'Reading video demographics from {video_demo_path}')
    video_demo = pd.read_csv(video_demo_path)
    video_demo.columns = ['actor_id', 'age', 'sex', 'race', 'ethnicity']

    train_infos, val_infos = [], []
    for val_idx, val_split in enumerate(paths):

        train_split = paths[:val_idx] + paths[val_idx + 1:]
        val_split = pd.read_csv(val_split, sep=' ', header=None)

        train_split = [pd.read_csv(split, sep=' ', header=None)
                       for split in train_split]
        train_split = pd.concat(train_split)

        train_paths = train_split[0].to_list()
        val_paths = val_split[0].to_list()

        train_info = pd.DataFrame(
            [get_info_from_filepath(path) for path in train_paths]).astype({'actor_id': 'int64'})
        val_info = pd.DataFrame([get_info_from_filepath(path)
                                for path in val_paths]).astype({'actor_id': 'int64'})

        train_info = pd.merge(train_info, video_demo,
                              how='inner', on='actor_id')
        val_info = pd.merge(val_info, video_demo, how='inner', on='actor_id')
        train_infos.append(train_info)
        val_infos.append(val_info)

    make_age_norm_plot(train_infos, val_infos)


def make_age_norm_plot(train_infos: List[pd.DataFrame], val_infos: List[pd.DataFrame]) -> None:
    train_infos = [train_info['age'] for train_info in train_infos]
    val_infos = [val_info['age'] for val_info in val_infos]
    train_mean_ages = [train_info.mean() for train_info in train_infos]
    val_mean_ages = [val_info.mean() for val_info in val_infos]

    train_mean_mean_age = sum(train_mean_ages) / len(train_mean_ages)
    val_mean_mean_age = sum(val_mean_ages) / len(val_mean_ages)

    train_std_mean_ages = (sum([((x - train_mean_mean_age) ** 2)
                           for x in train_mean_ages]) / len(train_mean_ages)) ** 0.5
    val_std_mean_ages = (sum([((x - val_mean_mean_age) ** 2)
                         for x in val_mean_ages]) / len(val_mean_ages)) ** 0.5

    plt.figure()

    x_axis = np.arange(0, 100, step=0.001)

    plt.xlabel('Age [a]')

    plt.plot(x_axis, scipy.stats.norm.pdf(
        x_axis, train_mean_mean_age, train_std_mean_ages), label='Training data')
    plt.plot(x_axis, scipy.stats.norm.pdf(
        x_axis, val_mean_mean_age, val_std_mean_ages), label='Validation data')
    
    plt.title('Distribution of mean ages within all training/validation splits.')
    plt.legend()

    plt.show()



def make_feature_bar_plot(train_infos, val_infos, key: str, label: str) -> None:

    train_features, val_features = [], []
    for train_info, val_info in zip(train_infos, val_infos):

        train_features.append(train_info[key].value_counts() / len(train_info))
        val_features.append(val_info[key].value_counts() / len(val_info))

    train_features = pd.DataFrame(train_features)
    val_features = pd.DataFrame(val_features)

    def _get_mean_and_std(features, key: str):
        mean = features[key].mean()
        std = features[key].std()

        return mean, std

    train_y, train_err = [], []
    val_y, val_err = [], []

    for feature in train_features.columns:
        mean, std = _get_mean_and_std(train_features, feature)
        train_y.append(mean)
        train_err.append(std)

    for feature in val_features.columns:
        mean, std = _get_mean_and_std(val_features, feature)
        val_y.append(mean)
        val_err.append(std)

    plt.figure()

    plt.xlabel(label)
    plt.ylabel('Percentage')

    plt.bar(train_features.columns, train_y, -0.2,
            align='edge', label='Training splits', yerr=train_err)
    plt.bar(val_features.columns, val_y, 0.2,
            align='edge', label='Validation splits', yerr=val_err)

    plt.legend()

    plt.show()


@cli.command()
@click.argument('root', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--k', default=5)
def run(root, output_dir, k):

    click.echo(f'Reading dataset at {click.format_filename(root)}')
    paths = list(sorted(glob.glob(os.path.join(root, 'VideoFlash', '*.flv'))))
    click.echo(f'Found a total of {len(paths)} videos.')

    infos = [get_info_from_filepath(path) for path in paths]
    actors = list(set(info['actor_id'] for info in infos))
    random.shuffle(actors)
    emotions_mapping = list(set(info['emotion'] for info in infos))
    emotions_mapping = {emotion: idx for idx,
                        emotion in enumerate(emotions_mapping)}

    df = pd.DataFrame([dict(path=path) | info for path,
                      info in zip(paths, infos)])

    template = Environment(loader=FileSystemLoader('./configs/templates'),
                           trim_blocks=True, lstrip_blocks=True).get_template('vith16_384_16x8x3.j2')

    actor_splits = np.array_split(actors, k)

    split_paths = list()
    for split_idx, actor_split in enumerate(actor_splits):

        split = df[df['actor_id'].isin(actor_split)][['path', 'emotion']]
        split['emotion'] = split['emotion'].apply(
            lambda x: emotions_mapping.get(x))
        output_path = get_split_file_path(
            output_dir=output_dir, split_idx=split_idx)
        split.to_csv(output_path, sep=' ', index=False, header=False)
        click.echo(f'Wrote split file to {output_path}')

        split_paths.append(output_path)

    for split_idx, val_split_path in enumerate(split_paths):

        config = template.render(
            dict(
                tag=f'crema_d_{split_idx}',
                train_split_paths=split_paths[:split_idx] +
                    split_paths[split_idx+1:],
                val_split_paths=[val_split_path]
            )
        )

        config_file_path = get_config_file_path(split_idx)
        with open(config_file_path, 'w') as config_file:
            config_file.write(config)
        click.echo(f'Wrote new config file to {config_file_path}')


@cli.command()
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--k', default=5)
def clean(output_dir, k) -> None:

    for split_num in range(k):
        config_file_path = get_config_file_path(split_num=split_num)
        os.remove(config_file_path)
        click.echo(f'Removed config file from {config_file_path}')

        train_split_path = get_split_file_path(
            output_dir=output_dir, split_num=split_num, split='train')
        os.remove(train_split_path)
        click.echo(f'Removed training split from {train_split_path}')

        val_split_path = get_split_file_path(output_dir, split_num, 'val')
        os.remove(val_split_path)
        click.echo(f'Removed validation split from {val_split_path}')


def get_config_file_path(split_num: int) -> str:
    return f'./configs/evals/vith16_384_crema_d_split_{split_num}_16x8x3.yaml'


def get_split_file_path(output_dir: str, split_idx: int) -> str:
    return os.path.join(output_dir, f'split_{split_idx}.csv')


def get_analysis_file_path(output_dir: str, split_num: int, split: Literal['train', 'val']) -> str:
    pass


def get_info_from_filepath(path: str):
    base_name = os.path.basename(path)
    file_name, _ = os.path.splitext(base_name)
    actor_id, sentence, emotion, intensity = file_name.split('_')
    return dict(
        actor_id=actor_id,
        sentence=sentence,
        emotion=emotion,
        intensity=intensity,
        path=path
    )


if __name__ == '__main__':
    cli()
