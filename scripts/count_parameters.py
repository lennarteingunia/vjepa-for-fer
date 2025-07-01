import os
import click
import torch
import yaml

from evals.pretrained_full_video_classification.eval import init_model, load_checkpoint
from models.attentive_pooler import AttentiveClassifier


@click.group('cli')
def cli() -> None:
    pass


@cli.command('count')
@click.argument('config', type=click.File('rb'))
def count(config) -> None:
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config = yaml.load(config, Loader=yaml.FullLoader)

    args_pretrain = config.get('pretrain')
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    tag = args_pretrain.get('write_tag', None)

    pretrained_path = os.path.join(pretrain_folder, ckp_fname)

    eval_tag = config.get('tag', None)

    folder = os.path.join(pretrain_folder, 'video_classification_frozen/')
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')

    args_opt = config.get('optimization')
    resolution = args_opt.get('resolution', 224)

    args_data = config.get('data')
    num_classes = args_data.get('num_classes')

    click.echo(f'Loading encoder model from {pretrained_path}')

    encoder = init_model(
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        crop_size=resolution,
        frames_per_clip=pretrain_frames_per_clip,
        tubelet_size=tubelet_size,
        use_sdpa=True,
        use_SiLU=True,
        tight_SiLU=True,
        uniform_power=True,
        checkpoint_key=checkpoint_key
    ).to(device)

    click.echo(f'Loaded encoder from {pretrained_path}')
    click.echo(f'Loading attentive classifier from {latest_path}')

    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,
        num_classes=num_classes,
    ).to(device)

    checkpoint = torch.load(latest_path, map_location=torch.device('cpu'))

    # -- loading encoder
    pretrained_dict = checkpoint['classifier']
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    classifier.load_state_dict(pretrained_dict)

    del checkpoint

    click.echo(f'Loaded attentive classifier from {latest_path}')

    encoder_num_params = sum(p.numel() for p in encoder.parameters())
    classifier_num_params = sum(p.numel() for p in classifier.parameters())
    total_num_params = encoder_num_params + classifier_num_params

    click.echo(
        f'Total number of encoder parameters are:\t{encoder_num_params}')
    click.echo(
        f'Total number of classifier parameters are:\t{classifier_num_params}')
    click.echo(f'Total number of parameters are:\t\t{total_num_params}')


if __name__ == '__main__':
    cli()
