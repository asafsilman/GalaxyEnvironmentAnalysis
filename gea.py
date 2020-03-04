import logging
from pathlib import Path

import click
from src.utils.load_config import load_config
from src.data.make_data_set import make_data_set
from src.data.split_data_set import split_data_set
from src.data.move_data import move_data
from src.models.train_model import train_model

logger = logging.getLogger(__name__)

def rm_tree(pth: Path, depth: int=0):
    for child in pth.iterdir():
        if child.is_file():
            logger.debug(f"Deleting file: {child}")
            child.unlink()
        else:
            rm_tree(child, depth+1)
    if depth > 0:
        logger.debug(f"Deleting directory: {pth}")
        pth.rmdir()

@click.group()
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def cli(ctx, debug):
    logging_level = logging.INFO
    if debug:
        logging_level = logging.DEBUG
    logging.basicConfig(format='%(levelname)s %(message)s', level=logging_level)

@cli.command()
@click.pass_context
def clean(ctx):
    processed_data = Path(__file__).parent.joinpath("data", "processed")  
    interim_data = Path(__file__).parent.joinpath("data", "interim")  
    
    logger.info(f'Cleaning {processed_data} directory')
    rm_tree(processed_data)

    logger.info(f'Cleaning {interim_data} directory')
    rm_tree(interim_data)


@cli.command()
@click.pass_context
@click.argument('config-file', type=click.Path(exists=True))
def data_prep(ctx, config_file):
    config = load_config(click.format_filename(config_file))
    make_data_set(config)
    
@cli.command()
@click.pass_context
@click.argument('config-file', type=click.Path(exists=True))
def data_split(ctx, config_file):
    config = load_config(click.format_filename(config_file))
    split_data_set(config)

@cli.command()
@click.pass_context
@click.argument('config-file', type=click.Path(exists=True))
@click.argument('directory', required=False, default="data/processed/test", type=click.Path())
def data_move(ctx, config_file, directory):
    config = load_config(click.format_filename(config_file))
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    move_data(config, directory)
    

@cli.command()
@click.pass_context
@click.argument('config-file', type=click.Path(exists=True))
def data_prep_split(ctx, config_file):
    config = load_config(click.format_filename(config_file))
    make_data_set(config)
    split_data_set(config)

@cli.command()
@click.pass_context
@click.argument('config-file', type=click.Path(exists=True))
@click.option('--new-model/--load-model', default=False)
@click.option('--save-training/--discard-training', default=True)
def train(ctx, config_file, new_model, save_training):
    config = load_config(click.format_filename(config_file))
    train_model(config, new_model, save_training)


if __name__ == '__main__':
    cli(obj={})