from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_beta_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_beta_sweep", SCRIPT_PATH)
assert SPEC and SPEC.loader
run_beta_sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_beta_sweep)


class BetaSweepTest(unittest.TestCase):
    def test_parse_accuracies_uses_aggregator_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "run.log"
            log_path.write_text(
                "client 0: local_test_acc=99.00%\n"
                "  aggregator: avg_acc=70.00% round_payload=0B\n"
                "  aggregator: avg_acc=80.00% round_payload=0B\n",
                encoding="utf-8",
            )
            self.assertEqual(
                run_beta_sweep.parse_accuracies(log_path),
                [70.0, 80.0],
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
                "status": "completed",
                "algorithm": "prototype",
                "beta": "0.5",
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

            run_beta_sweep.fill_accuracy_metrics(prototype, "local")
            run_beta_sweep.update_local_deltas([local, prototype])

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
                        "status": "completed",
                        "algorithm": algorithm,
                        "beta": "0.5",
                        "seed": str(seed),
                        "last10_avg_acc": str(score),
                        "final_avg_acc": str(score),
                        "best_avg_acc": str(score),
                        "delta_vs_local": "",
                    }
                )
        run_beta_sweep.update_local_deltas(tasks)

        with tempfile.TemporaryDirectory() as temporary:
            summary_path = Path(temporary) / "summary.csv"
            run_beta_sweep.write_summary(summary_path, tasks)
            with summary_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        prototype = next(row for row in rows if row["algorithm"] == "prototype")
        self.assertEqual(prototype["completed_runs"], "2")
        self.assertEqual(prototype["last10_acc_mean"], "77.000000")
        self.assertEqual(prototype["delta_vs_local_mean"], "6.000000")


if __name__ == "__main__":
    unittest.main()
