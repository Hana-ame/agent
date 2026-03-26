try:
    import typer
except ImportError:
    print("typer not installed, skipping")
    exit(0)

app = typer.Typer()

@app.command()
def hello(name: str = "World"):
    """Say hello."""
    typer.echo(f"Hello, {name}!")

@app.command()
def bye(count: int = 1):
    """Say goodbye."""
    for _ in range(count):
        typer.echo("Goodbye!")

if __name__ == "__main__":
    app()
