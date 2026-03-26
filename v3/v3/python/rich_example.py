try:
    from rich import print
    from rich.table import Table
    from rich.console import Console
except ImportError:
    print("rich not installed, skipping")
    exit(0)

console = Console()
console.print("[bold green]Hello[/] [red]World![/]")

table = Table(title="User Info")
table.add_column("Name", style="cyan")
table.add_column("Age", style="magenta")
table.add_row("Alice", "30")
table.add_row("Bob", "25")
console.print(table)
