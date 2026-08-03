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
    def test_bar_value_labels_use_metric_appropriate_precision(self) -> None:
        self.assertEqual(plot_experiments.bar_value_label(93.678, "accuracy"), "93.68")
        self.assertEqual(plot_experiments.bar_value_label(-1.234, "delta"), "-1.23")
        self.assertEqual(
            plot_experiments.bar_value_label(0.0078, "communication"),
            "0.008",
        )

    def test_parse_log_reads_spaced_accuracy_and_communication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.log"
            path.write_text(
                "  aggregator: avg_acc= 4.17% round_payload=0B\n"
                "  communication: round=4080 B (3.98 KiB) total=4080 B\n"
                "  fabric_traffic: round_rx=4000 B round_tx=5000 B "
                "round_total=9000 B (8.79 KiB) total=9000 B\n"
                "  adapter_traffic: round_http_rx=100 B round_http_tx=200 B "
                "round_grpc_rx=300 B round_grpc_tx=400 B round_total=1000 B total=1000 B\n"
                "  fabric_plus_adapter_traffic: round_total=10000 B total=10000 B\n"
                "  aggregator: avg_acc=80.00% round_payload=0B\n",
                encoding="utf-8",
            )
            (
                accuracies,
                uploads,
                downloads,
                communication,
                fabric_traffic,
                adapter_traffic,
                combined_traffic,
            ) = plot_experiments.parse_log(path)

        self.assertEqual(accuracies, (4.17, 80.0))
        self.assertEqual(uploads, (4080,))
        self.assertEqual(downloads, (4080,))
        self.assertEqual(communication, (8160,))
        self.assertEqual(fabric_traffic, (9000,))
        self.assertEqual(adapter_traffic, (1000,))
        self.assertEqual(combined_traffic, (10000,))

    def test_parse_log_reads_bidirectional_logical_communication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.log"
            path.write_text(
                "  logical_communication: upload=4080 B (3.98 KiB) "
                "download=2040 B (1.99 KiB) round_total=6120 B (5.98 KiB) "
                "total_upload=4080 B total_download=2040 B total=6120 B\n",
                encoding="utf-8",
            )
            _, uploads, downloads, communication, _, _, _ = plot_experiments.parse_log(path)

        self.assertEqual(uploads, (4080,))
        self.assertEqual(downloads, (2040,))
        self.assertEqual(communication, (6120,))

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
        self.assertEqual(runs[0].total_upload_bytes, 2040)
        self.assertEqual(runs[0].total_download_bytes, 2040)
        self.assertEqual(runs[0].total_communication_bytes, 4080)
        self.assertEqual(runs[0].total_endpoint_io_estimate_bytes, 8160)
        self.assertEqual(runs[0].comparison_communication_bytes, 8160)
        self.assertEqual(
            runs[0].comparison_communication_basis,
            "estimated_memory_endpoint_io",
        )
        self.assertEqual(runs[0].total_fabric_traffic_bytes, 0)

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
                "round_upload_bytes": (0,),
                "round_download_bytes": (0,),
                "round_communication_bytes": (0,),
                "round_fabric_traffic_bytes": (),
                "round_adapter_traffic_bytes": (),
                "round_fabric_plus_adapter_traffic_bytes": (),
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

    def test_fabric_comparison_prefers_adapter_inclusive_traffic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = plot_experiments.ExperimentRun(
                experiment_dir=root,
                task_id="fabric",
                dataset="mnist",
                model_config="homogeneous",
                partition="kn",
                partition_config="kn-test",
                beta="",
                samples_per_client="",
                ways="3",
                shots="100",
                stdev="2",
                train_shots_max="110",
                test_shots_per_class="15",
                algorithm="prototype_fabric",
                seed="1234",
                log_path=root / "run.log",
                accuracies=(90.0,),
                round_upload_bytes=(40800,),
                round_download_bytes=(40800,),
                round_communication_bytes=(81600,),
                round_fabric_traffic_bytes=(4_000_000,),
                round_adapter_traffic_bytes=(100_000,),
                round_fabric_plus_adapter_traffic_bytes=(4_100_000,),
            )

        self.assertEqual(run.total_endpoint_io_estimate_bytes, 163200)
        self.assertEqual(run.total_adapter_traffic_bytes, 100_000)
        self.assertEqual(run.comparison_communication_bytes, 4_100_000)
        self.assertEqual(
            run.comparison_communication_basis,
            "fabric_container_plus_adapter_io",
        )

        legacy = plot_experiments.ExperimentRun(
            **{
                **run.__dict__,
                "round_adapter_traffic_bytes": (),
                "round_fabric_plus_adapter_traffic_bytes": (),
            }
        )
        self.assertEqual(legacy.comparison_communication_bytes, 4_000_000)
        self.assertEqual(
            legacy.comparison_communication_basis,
            "fabric_container_rx_plus_tx_legacy",
        )


if __name__ == "__main__":
    unittest.main()
