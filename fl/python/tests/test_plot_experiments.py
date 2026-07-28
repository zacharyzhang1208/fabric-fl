from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "plot_experiments.py"
SPEC = importlib.util.spec_from_file_location("plot_experiments", SCRIPT_PATH)
assert SPEC and SPEC.loader
plot_experiments = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plot_experiments
SPEC.loader.exec_module(plot_experiments)


class PlotExperimentsTest(unittest.TestCase):
    def test_parse_log_reads_spaced_accuracy_and_communication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.log"
            path.write_text(
                "  aggregator: avg_acc= 4.17% round_payload=0B\n"
                "  communication: round=4080 B (3.98 KiB) total=4080 B\n"
                "  aggregator: avg_acc=80.00% round_payload=0B\n",
                encoding="utf-8",
            )
            accuracies, communication = plot_experiments.parse_log(path)

        self.assertEqual(accuracies, (4.17, 80.0))
        self.assertEqual(communication, (4080,))

    def test_load_runs_falls_back_to_task_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary)
            task_id = "kn-task"
            run_dir = experiment_dir / "runs" / task_id
            run_dir.mkdir(parents=True)
            log_path = run_dir / "run.log"
            log_path.write_text(
                "  aggregator: avg_acc=75.00% round_payload=0B\n"
                "  communication: round=2040 B total=2040 B\n",
                encoding="utf-8",
            )
            with (experiment_dir / "manifest.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "task_id",
                        "partition",
                        "partition_config",
                        "beta",
                        "samples_per_client",
                        "ways",
                        "shots",
                        "stdev",
                        "train_shots_max",
                        "test_shots_per_class",
                        "algorithm",
                        "seed",
                        "status",
                        "log_path",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": task_id,
                        "partition": "kn",
                        "partition_config": "kn-ways-3-shots-100-stdev-2",
                        "ways": "3",
                        "shots": "100",
                        "stdev": "2",
                        "algorithm": "prototype",
                        "seed": "1234",
                        "status": "completed",
                        "log_path": "/moved/path/run.log",
                    }
                )
            runs, warnings = plot_experiments.load_runs([experiment_dir])

        self.assertEqual(warnings, [])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].last10_accuracy, 75.0)
        self.assertEqual(runs[0].total_communication_bytes, 2040)

    def test_paired_deltas_match_partition_config_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = {
                "experiment_dir": root,
                "task_id": "task",
                "dataset": "mnist",
                "model_config": "homogeneous",
                "partition": "beta",
                "partition_config": "beta-0.5-samples-300",
                "beta": "0.5",
                "samples_per_client": "300",
                "ways": "",
                "shots": "",
                "stdev": "",
                "train_shots_max": "",
                "test_shots_per_class": "",
                "seed": "1234",
                "log_path": root / "run.log",
                "round_communication_bytes": (0,),
            }
            local = plot_experiments.ExperimentRun(
                algorithm="local",
                accuracies=(70.0,),
                **common,
            )
            prototype = plot_experiments.ExperimentRun(
                algorithm="prototype",
                accuracies=(76.5,),
                **common,
            )
            deltas = plot_experiments.paired_deltas([local, prototype])

        self.assertEqual(
            deltas[("beta-0.5-samples-300", "prototype")],
            [6.5],
        )


if __name__ == "__main__":
    unittest.main()
