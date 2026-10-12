import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_v26_experiment import (
    ABLATIONS,
    CITIES,
    ENDPOINTS,
    Experiment,
    _aggregate_rows,
)


class V26ExperimentRunnerTests(unittest.TestCase):
    def test_ablation_cells_cover_every_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Experiment(
                Path(tmp), "python", jobs=1, dry_run=True,
                seeds=(11, 12, 13),
            )
            cells = experiment.cells("ablation")

        self.assertEqual(
            len(cells),
            len(CITIES) * len(ENDPOINTS) * len(ABLATIONS) * 3,
        )
        self.assertEqual({cell[2] for cell in cells}, {11, 12, 13})

    def test_training_skips_only_after_completion_marker(self):
        class OneCellExperiment(Experiment):
            def __init__(self, root):
                super().__init__(root, "python", jobs=1, dry_run=False, seeds=(11,))
                self.calls = 0

            def cells(self, which):
                return [("nyc", ENDPOINTS[0], 11, "llm_numeric")]

            def _train_command(self, city, policy, seed, arm):
                return ["unused"]

            def _validate_numeric_cache(self, city, label="primary", expected_rows=20000):
                return {"rows": expected_rows}

            def _run(self, key, command, log, env=None):
                self.calls += 1
                prefix = self.prefix("nyc", ENDPOINTS[0]["label"], 11, "llm_numeric")
                Path(str(prefix) + "_run_config.json").parent.mkdir(parents=True, exist_ok=True)
                Path(str(prefix) + "_run_config.json").write_text("{}", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            experiment = OneCellExperiment(Path(tmp))
            experiment.train("main")
            experiment.train("main")
            marker = Path(str(experiment.prefix(
                "nyc", ENDPOINTS[0]["label"], 11, "llm_numeric"
            )) + "_complete.json")
            self.assertTrue(marker.exists())
            self.assertEqual(experiment.calls, 1)

    def test_training_selection_filters_city_and_policy(self):
        class RecordingExperiment(Experiment):
            def __init__(self, root):
                super().__init__(root, "python", jobs=1, dry_run=True, seeds=(11,))
                self.keys = []

            def _run(self, key, command, log, env=None):
                self.keys.append(key)

        artifact_root = Path(__file__).resolve().parent / "artifacts"
        artifact_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as tmp:
            experiment = RecordingExperiment(Path(tmp))
            experiment.train(
                "main",
                selected_cities=("boston",),
                selected_policies=("endpoint_profit",),
            )

        self.assertEqual(
            experiment.keys,
            ["train:boston:llm_numeric:endpoint_profit:seed11"],
        )

    def test_analyze_aggregates_rollouts_ood_baselines_and_external_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = Experiment(root, "python", jobs=1, dry_run=False, seeds=(11,))
            prefix = experiment.prefix("nyc", ENDPOINTS[0]["label"], 11, "llm_numeric")
            prefix.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "benchmark_evaluation": {
                    "rollout_summaries": {
                        "pi_price_gap@11": {
                            "objective_reward_mean": 1.0,
                            "objective_advantage_mean": 0.2,
                        }
                    }
                },
                "v26_ood_evaluation": {
                    "rollout_summaries": {
                        "pi_price_gap@perturbed0@11": {
                            "objective_reward_mean": 0.8,
                            "objective_advantage_mean": 0.1,
                        }
                    }
                },
                "v26_baseline_evaluation": {
                    "in_distribution_and_benchmarks": {
                        "comparison": {
                            "pi_price_gap@11": {
                                "rl_objective": 1.0,
                                "rl_minus_best_static": 0.2,
                                "rl_minus_myopic": 0.1,
                                "rl_minus_opening_hold": 0.3,
                            }
                        }
                    },
                    "ood_perturbed": {
                        "comparison": {
                            "pi_price_gap@perturbed0@11": {
                                "rl_objective": 0.8,
                                "rl_minus_best_static": 0.1,
                                "rl_minus_myopic": 0.05,
                                "rl_minus_opening_hold": 0.2,
                            }
                        }
                    },
                },
            }
            Path(str(prefix) + "_run_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            Path(str(prefix) + "_complete.json").write_text("{}", encoding="utf-8")
            external = root / "external" / "chicago"
            external.mkdir(parents=True)
            (external / "chicago_shared_trip_benchmark.json").write_text(
                json.dumps({
                    "benchmark": "test", "trips_used": 10, "pumas_used": 2,
                    "sources": {"llm_numeric": {
                        "spearman_predicted_vs_observed": 0.4,
                        "composition_only": {"predicted_logit_spread": 0.2},
                    }},
                }),
                encoding="utf-8",
            )

            with mock.patch(
                "scripts.run_v26_experiment._write_analysis_plots",
                return_value=["plot.png"],
            ):
                experiment.analyze()

            analyzed = root / "analyzed"
            with (analyzed / "v26_baseline_across_seeds.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                baseline = list(csv.DictReader(handle))
            with (analyzed / "v26_rollout_across_seeds.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rollout = list(csv.DictReader(handle))
            with (analyzed / "v26_external_benchmark.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                external_rows = list(csv.DictReader(handle))

        self.assertEqual({row["evaluation_set"] for row in baseline}, {
            "in_distribution_and_benchmarks", "ood_perturbed",
        })
        self.assertEqual(len(rollout), 2)
        self.assertEqual(external_rows[0]["source"], "llm_numeric")

    def test_aggregate_rows_reports_three_seed_interval(self):
        rows = [
            {"cell": "main", "seed": seed, "metric": value}
            for seed, value in zip((1, 2, 3), (1.0, 2.0, 3.0))
        ]
        aggregate = _aggregate_rows(rows, ("cell",), ("metric",))
        self.assertEqual(aggregate[0]["metric_n"], 3)
        self.assertAlmostEqual(aggregate[0]["metric_mean"], 2.0)
        self.assertLess(aggregate[0]["metric_lo"], 2.0)
        self.assertGreater(aggregate[0]["metric_hi"], 2.0)

    def test_numeric_training_declares_numeric_mode(self):
        artifact_root = Path(__file__).resolve().parent / "artifacts"
        artifact_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as tmp:
            experiment = Experiment(Path(tmp), "python", jobs=1, dry_run=True, seeds=(11,))
            command = experiment._train_command("nyc", ENDPOINTS[0], 11, "llm_numeric")
        index = command.index("--gpt_threshold_mode")
        self.assertEqual(command[index + 1], "llm_numeric")

    def test_numeric_cache_validation_rejects_fallback_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Experiment(Path(tmp), "python", jobs=1, dry_run=False, seeds=(11,))
            cache = experiment.numeric_cache("nyc")
            cache.parent.mkdir(parents=True)
            cache.write_text(json.dumps({
                "ProfileMarket": "New York City",
                "PriceThreshold": 2.0,
                "PriceThresholdSource": "fallback_rubric",
                "ThresholdPromptProtocol": "numeric_profile_threshold_v11",
                "ThresholdPromptModel": "gpt-4.1",
                "ThresholdRawLLMValue": 2.0,
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fallback/non-numeric"):
                experiment._validate_numeric_cache("nyc", expected_rows=1)


if __name__ == "__main__":
    unittest.main()
