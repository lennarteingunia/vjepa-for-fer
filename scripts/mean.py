import click
import numpy as np
import yaml


@click.group('cli')
def cli():
    pass


@cli.command('mean')
@click.argument('inputs', type=click.File('rb'), nargs=-1)
@click.argument('output', type=click.Path(resolve_path=True), nargs=1)
@click.option('--key', multiple=True, required=True)
def mean(inputs, output, key) -> None:
    metrics = [yaml.load(input, Loader=yaml.FullLoader) for input in inputs]
    metrics = {k: np.array([metric[k] for metric in metrics]
                           ).mean().item() for k in list(key)}
    with open(output, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    click.echo(f'Wrote output file to {output}')


if __name__ == '__main__':
    cli()
