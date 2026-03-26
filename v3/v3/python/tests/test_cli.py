import pytest
from click.testing import CliRunner
from myproject.cli import main

def test_main():
    runner = CliRunner()
    result = runner.invoke(main, ['--name', 'Alice'])
    assert result.exit_code == 0
    assert 'Hello, Alice!' in result.output
