"""Command-line interface."""
import click

@click.command()
@click.option('--name', default='World', help='Who to greet')
def main(name):
    """Simple greeting CLI."""
    click.echo(f"Hello, {name}!")

if __name__ == '__main__':
    main()
