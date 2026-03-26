try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False
    print("click not installed, skipping demo")
    exit(0)

@click.command()
@click.option('--name', default='World', help='Who to greet')
@click.option('--count', default=1, type=int, help='Number of greetings')
def greet(name, count):
    for _ in range(count):
        click.echo(f"Hello, {name}!")

if __name__ == "__main__":
    greet(['--name', 'Click', '--count', '2'])
