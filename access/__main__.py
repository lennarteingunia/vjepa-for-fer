import logging
import click
import torch

from vjepa import build_attentive_classifier, build_encoder


@click.group('cli')
def cli():
    click.echo('You started the V-JEPA command line utility.')


@cli.command('check_checkpoint')
@click.argument('config')
@click.argument('checkpoint')
@click.option('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
def check_checkpoint(config: str, checkpoint: str, *, device: str) -> None:
    logging.basicConfig(level=logging.INFO)
    device = torch.device(device)
    click.echo(
        f'Checking wether the attentive classifier can be loaded from the checkpoint using:\nConfig:\t\t{config}\nCheckpoint:\t{checkpoint}')
    attentive_classifier = build_attentive_classifier(
        config=config,
        checkpoint_path=checkpoint,
        device=device
    )
    encoder = build_encoder(
        config=config,
        checkpoint_path=checkpoint,
        device=device
    )


cli()
