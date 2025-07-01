import argparse
import itertools
import logging
import os
from typing import List, Literal, Optional, Union

import click
import numpy
from sklearn.decomposition import PCA
import torch
import tqdm
import yaml
from datasets.full_video_dataset import FullVideoDataset
from evals.pretrained_full_video_classification.eval import init_model
from evals.video_classification_frozen.utils import make_transforms
from evals.video_classification_frozen_full_video.utils import ClipAggregation, FrameAggregation
from models.attentive_pooler import AttentiveClassifier

from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class EmbedWrap(AttentiveClassifier):

    def forward(self, x):
        return self.pooler(x).squeeze(1)


def make_fullvideodata(
    data_paths: List[str],
    batch_size: int,
    frames_per_clip: int,
    frame_step: int,
    transform,
    *,
    num_workers: int = 10,
    force_indexing: bool = True,
    disable_indexing_tqdm: bool = True,
    logger: logging.Logger = logging.getLogger(__name__)
):

    dataset = FullVideoDataset(
        data_paths=data_paths,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        transform=transform,
        logger=logger,
        min_pad=True,
        force_indexing=force_indexing,
        disable_tqdm=disable_indexing_tqdm,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    return dataset, data_loader


def make_dataloader(
    root_path,
    batch_size,
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_views_per_segment=1,
    training=False,
    num_workers=12,
):
    # Make Video Transforms
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4/3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=resolution,
    )

    _dataset, data_loader = make_fullvideodata(
        data_paths=root_path,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        transform=transform,
        num_workers=num_workers,
        logger=logger
    )

    return data_loader


def make_to_file(
    args_eval: dict[str, dict],
    outputdir: str,
) -> None:

    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- PRETRAIN
    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- DATA
    args_data = args_eval.get('data')
    val_data_path = args_data.get('dataset_val')
    if not (type(val_data_path) is list):
        val_data_path = [val_data_path]
    num_classes = args_data.get('num_classes')
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_frame_step = args_pretrain.get('frame_step', 4)

    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    batch_size = args_opt.get('batch_size')
    attend_across_segments = args_opt.get('attend_across_segments', False)

    eval_tag = args_eval.get('tag', None)

    # ----------------------------------------------------------------------- #

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, 'video_classification_frozen/')
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')

    # -- pretrained encoder (frozen)
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa)

    if pretrain_frames_per_clip == 1:
        # Process each frame independently and aggregate
        encoder = FrameAggregation(encoder).to(device)
    else:
        # Process each video clip independently and aggregate
        encoder = ClipAggregation(
            encoder,
            tubelet_size=tubelet_size,
            attend_across_segments=attend_across_segments
        ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # -- init classifier
    classifier = EmbedWrap(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,
        num_classes=num_classes,
    ).to(device)

    val_loader = make_dataloader(
        root_path=val_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        batch_size=batch_size,
    )

    ipe = len(val_loader)

    logger.info(f'Dataloader created... iterations per epoch: {ipe}')

    classifier = load_checkpoint(
        r_path=latest_path,
        classifier=classifier,
    )

    classifier.train(mode=False)
    gax = []
    x = []
    y = []
    for iteration, data in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):

            # Load data and put on GPU
            clips = [
                [dij.to(device, non_blocking=True)
                 for dij in di]  # iterate over spatial views of clip
                for di in data[0]  # iterate over temporal index of clip
            ]
            clip_indices = [d.to(device, non_blocking=True)
                            for d in data[2]['indices']]
            labels = data[1].to(device)

            video_paths = data[2]['path']

            # Forward and prediction
            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
                gax_outputs = [o.mean(dim=1) for o in outputs]
                outputs = [classifier(o) for o in outputs]
                x.extend([xs.detach().cpu() for xs in outputs])
                gax.extend([gaxs.detach().cpu() for gaxs in gax_outputs])
                y.extend([ys.detach().cpu() for ys in labels])

    gax = torch.cat(gax, dim=0).numpy()
    x = torch.cat(x, dim=0).numpy()
    y = torch.stack(y, dim=0).numpy()

    numpy.save(os.path.join(outputdir, "gax.npy"), gax)
    numpy.save(os.path.join(outputdir, "x.npy"), x)
    numpy.save(os.path.join(outputdir, "y.npy"), y)

    with open(os.path.join(outputdir, "config.yaml"), "w")as f:
        yaml.dump(args_eval, f, default_flow_style=False)


def show_and_save_plot(
    x: numpy.ndarray,
    y: numpy.ndarray,
    perplexity: float = 30,
    early_exaggeration: float = 12,
    learning_rate: Union[float, Literal["auto"]] = "auto",
    n_iter: int = 1000,
    n_iter_without_progress: int = 300,
    min_grad_norm: float = 1e-7,
    save_path: Optional[os.PathLike] = None,
):

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        n_iter=n_iter,
        n_iter_without_progress=n_iter_without_progress,
        min_grad_norm=min_grad_norm,
    )
    tsne_result = tsne.fit_transform(x)
    tsne_result_df = pd.DataFrame(
        dict(
            tsne_1=tsne_result[:, 0],
            tsne_2=tsne_result[:, 1],
            label=y,
        )
    )
    _, ax = plt.subplots(1)
    sns.scatterplot(
        x='tsne_1',
        y='tsne_2',
        hue='label',
        data=tsne_result_df,
        ax=ax,
        s=120,
        palette="deep"
    )
    lim = (tsne_result.min() - 5, tsne_result.max() + 5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(
        title="Labels",
        bbox_to_anchor=(1.05, 1),
        loc=2, borderaxespad=0.0,
        labels=["Sad", "Fear", "Happy", "Neutral", "Disgust", "Anger"]
    )

    if save_path:

        plt.savefig(save_path)

    plt.show()


def load_checkpoint(
    r_path,
    classifier,
):
    checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']

    # -- loading encoder
    pretrained_dict = checkpoint['classifier']
    pretrained_dict = {key.replace(
        'module.', ''): value for key, value in pretrained_dict.items()}
    msg = classifier.load_state_dict(pretrained_dict)
    logger.info(
        f'loaded pretrained classifier from epoch {epoch} with msg: {msg}')

    del checkpoint

    return classifier


@click.group('cli')
def cli():
    click.echo("Starting a t-SNE command line tool")


@cli.command()
@click.argument("configfile", type=click.Path(exists=True))
@click.argument("outputdir")
def make(configfile, outputdir):
    with open(configfile, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    make_to_file(
        args_eval=config,
        outputdir=outputdir
    )


@cli.command()
@click.argument("inputdir")
@click.option("--min-perplexity", required=False, default=30)
@click.option("--max-perplexity", required=False, default=100)
@click.option("--step-perplexity", required=False, default=10)
@click.option("--early-exaggeration", required=False, default=12)
@click.option("--min-learning-rate", required=False, default=1)
@click.option("--max-learning-rate", required=False, default=20)
@click.option("--step-learning-rate", required=False, default=5)
@click.option("--n-iter", required=False, default=3000)
@click.option("--n-iter-without-progress", required=False, default=300)
@click.option("--min-grad-norm", required=False, default=1e-7)
def plot(
    inputdir,
    min_perplexity,
    max_perplexity,
    step_perplexity,
    early_exaggeration,
    min_learning_rate,
    max_learning_rate,
    step_learning_rate,
    n_iter,
    n_iter_without_progress,
    min_grad_norm
):
    gax = numpy.load(os.path.join(inputdir, "gax.npy"))
    x = numpy.load(os.path.join(inputdir, "x.npy"))
    y = numpy.load(os.path.join(inputdir, "y.npy"))

    perplexity_it = range(min_perplexity, max_perplexity, step_perplexity)
    learning_rate_it = range(
        min_learning_rate, max_learning_rate, step_learning_rate)

    for perplexity, learning_rate in itertools.product(*[perplexity_it, learning_rate_it]):

        gax_save_path = os.path.join(
            inputdir, f"gax_p{perplexity}_eex{early_exaggeration}_lr{learning_rate}_it{n_iter}.svg")
        attp_save_path = os.path.join(
            inputdir, f"attp_p{perplexity}_eex{early_exaggeration}_lr{learning_rate}_it{n_iter}.svg")

        show_and_save_plot(
            gax,
            y,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            n_iter=n_iter,
            n_iter_without_progress=n_iter_without_progress,
            min_grad_norm=min_grad_norm,
            save_path=gax_save_path
        )

        show_and_save_plot(
            x=x,
            y=y,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            n_iter=n_iter,
            n_iter_without_progress=n_iter_without_progress,
            min_grad_norm=min_grad_norm,
            save_path=attp_save_path
        )


@cli.command()
@click.argument("input_dir")
def pca(input_dir):

    gax = numpy.load(os.path.join(input_dir, "gax.npy"))
    x = numpy.load(os.path.join(input_dir, "x.npy"))
    y = numpy.load(os.path.join(input_dir, "y.npy"))

    x_save_path = os.path.join(input_dir, "attp_pca.svg")
    gax_save_path = os.path.join(input_dir, "gax_pca.svg")

    show_and_save_pca_plot(x, y, x_save_path)
    show_and_save_pca_plot(gax, y, gax_save_path)


def show_and_save_pca_plot(x: numpy.ndarray, y: numpy.ndarray, save_path: os.PathLike) -> None:

    pca = PCA(n_components=2)
    pca_x = pca.fit_transform(x)

    pca_result_df = pd.DataFrame(
        dict(
            pca_1=pca_x[:, 0],
            pca_2=pca_x[:, 1],
            label=y,
        )
    )
    
    _, ax = plt.subplots(1)
    sns.scatterplot(
        x='pca_1',
        y='pca_2',
        hue='label',
        data=pca_result_df,
        ax=ax,
        s=120,
        palette="deep"
    )
    
    lim = (pca_x.min() - 5, pca_x.max() + 5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(
        title="Labels",
        bbox_to_anchor=(1.05, 1),
        loc=2, borderaxespad=0.0,
        labels=["Sad", "Fear", "Happy", "Neutral", "Disgust", "Anger"]
    )

    if save_path:

        plt.savefig(save_path)

    plt.show()


@cli.command()
@click.argument("configfile", type=click.Path(exists=True))
@click.argument("outputdir")
@click.pass_context
def run(ctxt, configfile, outputdir):
    ctxt.invoke(make, configfile=configfile, outputdir=outputdir)
    ctxt.invoke(plot, inputdir=outputdir)


if __name__ == '__main__':
    cli()
