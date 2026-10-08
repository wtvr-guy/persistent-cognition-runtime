def main() -> None:
    from persistent_cognition.cli import main as cli_main
    from persistent_cognition.diagnostics import run_command

    run_command(cli_main)
