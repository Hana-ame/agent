try:
    import click
except ImportError:
    print("click not installed, skipping")
    exit(0)

@click.command()
@click.option('--name', default='World', help='Who to greet')
@click.option('--count', default=1, type=int, help='Number of times to greet')
@click.option('--uppercase/--no-uppercase', default=False, help='Uppercase output')
def greet(name, count, uppercase):
    for i in range(count):
        msg = f"Hello, {name}!"
        if uppercase:
            msg = msg.upper()
        click.echo(msg)

if __name__ == '__main__':
    greet()
