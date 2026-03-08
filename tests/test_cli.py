from click.testing import CliRunner
from nexagen.cli.app import main


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "nexagen" in result.output


def test_run_help():
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--provider" in result.output
    assert "--tools" in result.output
    assert "--permission-mode" in result.output


def test_chat_help():
    runner = CliRunner()
    result = runner.invoke(main, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--provider" in result.output
