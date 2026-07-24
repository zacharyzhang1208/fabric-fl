from __future__ import annotations

import csv
import importlib.util
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_experiments.py"
SPEC = importlib.util.spec_from_file_location("run_experiments", SCRIPT_PATH)
assert SPEC and SPEC.loader
run_experiments = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_experiments)


def partition_fields(
    partition: str = "beta",
    config: str = "beta-0.5",
) -> dict[str, str]:
    return {
        "partition": partition,
        "partition_config": (
            "beta-0.5-samples-300"
            if config == "beta-0.5"
            else config
        ),
        "beta": "0.5" if partition == "beta" else "",
        "samples_per_client": "300" if partition == "beta" else "",
        "ways": "3" if partition == "kn" else "",
        "shots": "100" if partition == "kn" else "",
        "stdev": "2" if partition == "kn" else "",
        "train_shots_max": "110" if partition == "kn" else "",
        "test_shots_per_class": "15" if partition == "kn" else "",
    }


def runner_args(partition: str) -> Namespace:
    return Namespace(
        partition=partition,
        betas=[0.5, 0.2] if partition == "beta" else None,
        ways=3,
        shots=100,
        stdev=2,
        train_shots_max=110,
        test_shots_per_class=15,
        python="python3",
        dataset="mnist",
        data_dir="fl/data",
        samples_per_client=300,
        num_clients=20,
        model_config="homogeneous",
        rounds=10,
        local_epochs=1,
        batch_size=4,
        eval_batch_size=256,
        test_limit=300,
        lr=0.01,
        optimizer="sgd",
        proto_weight=0.5,
        proto_temperature=0.1,
        prototypes_per_class=1,
        min_samples_per_prototype=10,
        fedprox_mu=0.01,
        seeds=[1234],
        algorithms=["local", "prototype"],
    )


class ExperimentRunnerTest(unittest.TestCase):
    def test_beta_tasks_expand_values_algorithms_and_seeds(self) -> None:
        args = runner_args("beta")
        with tempfile.TemporaryDirectory() as temporary:
            tasks = run_experiments.make_tasks(args, Path(temporary))

        self.assertEqual(len(tasks), 4)
        self.assertEqual(
            {task["partition_config"] for task in tasks},
            {"beta-0.5-samples-300", "beta-0.2-samples-300"},
        )
        command = json_command(tasks[0])
        self.assertEqual(command[command.index("--partition") + 1], "beta")
        self.assertIn("--beta", command)
        self.assertIn("--samples-per-client", command)
        self.assertNotIn("--ways", command)

    def test_kn_tasks_use_kn_parameters_without_beta(self) -> None:
        args = runner_args("kn")
        with tempfile.TemporaryDirectory() as temporary:
            tasks = run_experiments.make_tasks(args, Path(temporary))

        self.assertEqual(len(tasks), 2)
        self.assertTrue(
            all(
                task["partition_config"] == "kn-ways-3-shots-100-stdev-2"
                "-trainmax-110-testshots-15"
                for task in tasks
            )
        )
        command = json_command(tasks[0])
        self.assertEqual(command[command.index("--partition") + 1], "kn")
        self.assertEqual(command[command.index("--ways") + 1], "3")
        self.assertEqual(command[command.index("--shots") + 1], "100")
        self.assertNotIn("--beta", command)
        self.assertNotIn("--samples-per-client", command)

    def test_parse_accuracies_uses_aggregator_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "run.log"
            log_path.write_text(
                "client 0: local_test_acc=99.00%\n"
                "  aggregator: avg_acc= 4.17% round_payload=0B\n"
                "  aggregator: avg_acc=80.00% round_payload=0B\n",
                encoding="utf-8",
            )
            self.assertEqual(
                run_experiments.parse_accuracies(log_path),
                [4.17, 80.0],
            )

    def test_metrics_and_paired_local_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "prototype.log"
            lines = [
                f"  aggregator: avg_acc={value:.2f}% round_payload=0B"
                for value in range(1, 13)
            ]
            log_path.write_text("\n".join(lines), encoding="utf-8")
            prototype = {
                **partition_fields(),
                "status": "completed",
                "algorithm": "prototype",
                "seed": "1234",
                "log_path": str(log_path),
                "last10_avg_acc": "",
                "final_avg_acc": "",
                "best_avg_acc": "",
                "delta_vs_local": "",
            }
            local = {
                **prototype,
                "algorithm": "local",
                "last10_avg_acc": "5.500000",
            }

            run_experiments.fill_accuracy_metrics(prototype)
            run_experiments.update_local_deltas([local, prototype])

            self.assertEqual(prototype["last10_avg_acc"], "7.500000")
            self.assertEqual(prototype["final_avg_acc"], "12.000000")
            self.assertEqual(prototype["best_avg_acc"], "12.000000")
            self.assertEqual(prototype["delta_vs_local"], "2.000000")

    def test_summary_groups_seeds(self) -> None:
        tasks = []
        for algorithm, scores in (("local", (70.0, 72.0)), ("prototype", (75.0, 79.0))):
            for seed, score in enumerate(scores):
                tasks.append(
                    {
                        **partition_fields(),
                        "status": "completed",
                        "algorithm": algorithm,
                        "seed": str(seed),
                        "last10_avg_acc": str(score),
                        "final_avg_acc": str(score),
                        "best_avg_acc": str(score),
                        "delta_vs_local": "",
                    }
                )
        run_experiments.update_local_deltas(tasks)

        with tempfile.TemporaryDirectory() as temporary:
            summary_path = Path(temporary) / "summary.csv"
            run_experiments.write_summary(summary_path, tasks)
            with summary_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        prototype = next(row for row in rows if row["algorithm"] == "prototype")
        self.assertEqual(prototype["completed_runs"], "2")
        self.assertEqual(prototype["last10_acc_mean"], "77.000000")
        self.assertEqual(prototype["delta_vs_local_mean"], "6.000000")

    def test_summary_keeps_beta_and_kn_groups_separate(self) -> None:
        tasks = []
        for fields in (
            partition_fields(),
            partition_fields(
                "kn",
                "kn-ways-3-shots-100-stdev-2-trainmax-110-testshots-15",
            ),
        ):
            tasks.append(
                {
                    **fields,
                    "status": "completed",
                    "algorithm": "local",
                    "seed": "1234",
                    "last10_avg_acc": "80",
                    "final_avg_acc": "80",
                    "best_avg_acc": "80",
                    "delta_vs_local": "0",
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            summary_path = Path(temporary) / "summary.csv"
            run_experiments.write_summary(summary_path, tasks)
            with summary_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual({row["partition"] for row in rows}, {"beta", "kn"})
        kn_row = next(row for row in rows if row["partition"] == "kn")
        self.assertEqual(kn_row["ways"], "3")
        self.assertEqual(kn_row["beta"], "")


def json_command(task: dict[str, str]) -> list[str]:
    import json

    return json.loads(task["command"])


if __name__ == "__main__":
    unittest.main()
