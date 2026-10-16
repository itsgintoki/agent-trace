from agent_trace.cli.main import main


def test_cli_list():
    assert main(["list"]) == 0


def test_cli_run_all_passes(capsys):
    assert main(["run"]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out


def test_cli_run_single(tmp_path):
    out = tmp_path / "report.json"
    assert main(["run", "--scenario", "lookup-query", "--output", str(out)]) == 0
    assert out.exists()


def test_cli_unknown_scenario():
    assert main(["run", "--scenario", "nope"]) == 2
