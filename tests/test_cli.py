from __future__ import annotations

import json
from pathlib import Path

from drdfl.cli import main


def test_cli_end_to_end(tmp_path: Path) -> None:
    generated = tmp_path / "generated.json"
    assert (
        main(
            [
                "generate",
                "--layers",
                "2",
                "--width",
                "2",
                "--context-dim",
                "3",
                "--seed",
                "5",
                "--output",
                str(generated),
            ]
        )
        == 0
    )
    assert json.loads(generated.read_text())["exact_shortest_path"]["objective"] > 0.0

    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    assert (
        main(
            [
                "collect",
                "--count",
                "8",
                "--layers",
                "2",
                "--width",
                "2",
                "--context-dim",
                "3",
                "--seed",
                "10",
                "--output",
                str(train),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "collect",
                "--count",
                "4",
                "--layers",
                "2",
                "--width",
                "2",
                "--context-dim",
                "3",
                "--seed",
                "20",
                "--output",
                str(validation),
            ]
        )
        == 0
    )
    oracle = tmp_path / "oracle.json"
    assert main(["oracle", str(validation), "--output", str(oracle)]) == 0
    assert json.loads(oracle.read_text())["comparison"]["verified"]

    checkpoint = tmp_path / "model.safetensors"
    training_report = tmp_path / "training.json"
    assert (
        main(
            [
                "train",
                str(train),
                "--validation",
                str(validation),
                "--mode",
                "spo_kl_dro",
                "--epochs",
                "1",
                "--warm-start-epochs",
                "0",
                "--hidden-dim",
                "8",
                "--hidden-layers",
                "1",
                "--seed",
                "30",
                "--checkpoint",
                str(checkpoint),
                "--output-report",
                str(training_report),
            ]
        )
        == 0
    )
    benchmark = tmp_path / "benchmark.json"
    benchmark_csv = tmp_path / "benchmark.csv"
    assert (
        main(
            [
                "benchmark",
                str(validation),
                "--train-dataset",
                str(train),
                "--checkpoint",
                f"spo_kl_dro={checkpoint}",
                "--bootstrap-draws",
                "10",
                "--output-json",
                str(benchmark),
                "--output-csv",
                str(benchmark_csv),
            ]
        )
        == 0
    )
    assert benchmark.exists() and benchmark_csv.exists()
