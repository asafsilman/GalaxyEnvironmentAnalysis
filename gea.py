from pathlib import Path

import click
from src.utils.load_config import load_config

@click.group()
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def cli(ctx, debug):
    ctx.obj['DEBUG'] = debug

@cli.command()
@click.pass_context
def clean(ctx):
    processed_data = Path(__file__).parent.joinpath("data", "processed")
    debug = ctx.obj['DEBUG']
    
    def rm_tree(pth: Path, depth: int, debug: bool):
        for child in pth.iterdir():
            if child.is_file():
                if debug: click.echo(f"Deleting file: {child}")
                child.unlink()
            else:
                rm_tree(child, depth+1, debug)
        if depth > 0:
            if debug: click.echo(f"Deleting directory: {child}")
            pth.rmdir()
    
    rm_tree(processed_data, 0, debug)

if __name__ == '__main__':
    cli(obj={})