try:
    import click
except ImportError:
    print("click not installed, skipping")
    exit(0)

@click.group()
def cli():
    """A group of commands."""
    pass

@cli.command()
@click.option('--name', default='World')
def hello(name):
    click.echo(f"Hello, {name}!")

@cli.command()
@click.option('--count', default=1, type=int)
def bye(count):
    for _ in range(count):
        click.echo("Goodbye!")

if __name__ == '__main__':
    cli()
