"""Backward-compat entry point: python -m ejepa_cli.main."""

from ejepa_cli.cli import app

if __name__ == "__main__":
    app()
