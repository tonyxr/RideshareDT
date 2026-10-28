import csv
import os
import tempfile
import unittest

import Core as core_module
from Core import Core, _infer_airport_trip, _infer_hour_day, _iter_tabular_rows
from model_registry import (
    allocate_archive_path,
    append_manifest,
    resolve_model_reference,
)


class ModelRegistryAndDatasetValidationTests(unittest.TestCase):
    def test_mixed_objective_artifact_round_trip_preserves_identity_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "mixed-policy.pt")
            objective = "profit_70_competitiveness_30"
            trained = Core(
                market_name="New York City",
                seed=37,
                choice_mode="parametric",
                firm1_mode="RL",
                firm2_mode="static",
                policy_objective=objective,
                total_customers_pool=20,
            )
            trained.save_trained_model(model_path, model_id="mixed-policy")
            artifact = core_module.torch.load(
                model_path, map_location="cpu", weights_only=False
            )
            active = artifact["mdp_config"]["active_reward"]
            self.assertEqual(artifact["policy_objective"], objective)
            self.assertEqual(
                artifact["reward_mechanism"],
                "economic_profit_competitiveness_mix_v25",
            )
            # 0.7 x profit endpoint (0.80 own, 0.20 lead) + 0.3 x competitiveness
            # endpoint (0.10 own, 0.20 own share, 0.70 share lead).
            self.assertAlmostEqual(active["own_profit_weight"], 0.59)
            self.assertAlmostEqual(active["profit_advantage_weight"], 0.14)
            self.assertAlmostEqual(active["market_share_level_weight"], 0.06)
            self.assertAlmostEqual(
                active["market_share_competitiveness_weight"], 0.21
            )

            matching = Core(
                market_name="New York City",
                seed=37,
                choice_mode="parametric",
                firm1_mode="RL",
                firm2_mode="static",
                policy_objective=objective,
                total_customers_pool=20,
            )
            matching.load_trained_model(model_path)
            mismatched = Core(
                market_name="New York City",
                seed=37,
                choice_mode="parametric",
                firm1_mode="RL",
                firm2_mode="static",
                policy_objective="profit_50_competitiveness_50",
                total_customers_pool=20,
            )
            with self.assertRaisesRegex(ValueError, "objective does not match"):
                mismatched.load_trained_model(model_path)

    def test_manipulated_coefficient_scope_is_persisted_and_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "three-coefficient-policy.pt")
            trained = Core(
                market_name="New York City",
                seed=29,
                choice_mode="parametric",
                firm1_mode="RL",
                firm2_mode="static",
                manipulated_price_coefficients=3,
                total_customers_pool=20,
            )
            trained.save_trained_model(model_path, model_id="three-coefficients")
            artifact = core_module.torch.load(
                model_path, map_location="cpu", weights_only=False
            )

            self.assertEqual(artifact["manipulated_price_coefficient_count"], 3)
            self.assertEqual(
                artifact["opt_keys"],
                ["base_fare", "per_minute", "per_mile"],
            )

            matching = Core(
                market_name="New York City",
                seed=29,
                choice_mode="parametric",
                firm1_mode="RL",
                firm2_mode="static",
                manipulated_price_coefficients=3,
                total_customers_pool=20,
            )
            matching.load_trained_model(model_path)

            mismatched = Core(
                market_name="New York City",
                seed=29,
                choice_mode="parametric",
                firm1_mode="RL",
                firm2_mode="static",
                manipulated_price_coefficients=2,
                total_customers_pool=20,
            )
            with self.assertRaisesRegex(
                ValueError, "count does not match runtime"
            ):
                mismatched.load_trained_model(model_path)

    def test_cached_population_contract_is_persisted_and_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = os.path.join(directory, "nyc-thresholds.jsonl")
            model_path = os.path.join(directory, "cached-policy.pt")
            with open(cache_path, "w", encoding="utf-8") as handle:
                handle.write(
                    '{"PriceThreshold": 2.0, "ProfileMarket": "New York City"}\n'
                )

            trained = Core(
                market_name="New York City",
                seed=31,
                choice_mode="parametric",
                firm1_mode="RL",
                firm2_mode="static",
                total_customers_pool=20,
                threshold_profile_source="cached",
                threshold_cache_path=cache_path,
            )
            trained.save_trained_model(model_path, model_id="cached-policy")
            artifact = core_module.torch.load(
                model_path, map_location="cpu", weights_only=False
            )
            contract = artifact["case_study_population"]
            self.assertEqual(artifact["version"], 23)
            self.assertEqual(artifact["manipulated_price_coefficient_count"], 5)
            self.assertEqual(
                artifact["reward_mechanism"],
                "economic_profit_focus_v22",
            )
            self.assertEqual(
                artifact["action_semantics"],
                "absolute_whole_tariff_target",
            )
            self.assertEqual(
                artifact["architecture"],
                "separate-gru-actor-critic-categorical-ppo",
            )
            self.assertEqual(
                artifact["competition_protocol"]["name"],
                "firm2_leader_then_firm1_follower",
            )
            self.assertEqual(
                artifact["competition_protocol"][
                    "shared_action_interval_steps"
                ],
                5,
            )
            self.assertEqual(
                artifact["competition_protocol"]["steps_per_day"],
                10,
            )
            self.assertNotIn("objective_q_update_calls", artifact)
            self.assertEqual(artifact["response_dim"], 0)
            self.assertEqual(artifact["opponent_forecast_dim"], 0)
            self.assertEqual(contract["market_name"], "New York City")
            self.assertEqual(contract["threshold_profile_source"], "cached")
            self.assertTrue(contract["threshold_cache_sha256"])

            wrong_objective = Core(
                market_name="New York City",
                seed=31,
                choice_mode="parametric",
                firm1_mode="RL",
                firm2_mode="static",
                policy_objective="competitiveness",
                total_customers_pool=20,
                threshold_profile_source="cached",
                threshold_cache_path=cache_path,
            )
            self.assertEqual(
                wrong_objective.long_term_profit_reward_config.own_profit_weight,
                0.10,
            )
            self.assertEqual(
                wrong_objective.long_term_profit_reward_config.market_share_competitiveness_weight,
                0.70,
            )
            with self.assertRaisesRegex(ValueError, "objective does not match"):
                wrong_objective.load_trained_model(model_path)

            wrong_source = Core(
                market_name="New York City",
                seed=31,
                choice_mode="parametric",
                firm1_mode="RL",
                firm2_mode="static",
                total_customers_pool=20,
                threshold_profile_source="generated",
            )
            with self.assertRaisesRegex(ValueError, "threshold population"):
                wrong_source.load_trained_model(model_path)

            with open(cache_path, "a", encoding="utf-8") as handle:
                handle.write(
                    '{"PriceThreshold": 3.0, "ProfileMarket": "New York City"}\n'
                )
            wrong_cache = Core(
                market_name="New York City",
                seed=31,
                choice_mode="parametric",
                firm1_mode="RL",
                firm2_mode="static",
                total_customers_pool=20,
                threshold_profile_source="cached",
                threshold_cache_path=cache_path,
            )
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                wrong_cache.load_trained_model(model_path)

    def test_registry_resolves_old_model_id_and_latest_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            first = allocate_archive_path(directory, "run-a")
            second = allocate_archive_path(directory, "run-b")
            open(first, "wb").close()
            open(second, "wb").close()
            append_manifest(directory, {
                "model_id": "run-a",
                "archive_id": os.path.basename(first),
                "archive_path": first,
            })
            append_manifest(directory, {
                "model_id": "run-b",
                "archive_id": os.path.basename(second),
                "archive_path": second,
            })

            self.assertEqual(resolve_model_reference("run-a", directory), first)
            self.assertEqual(resolve_model_reference("latest", directory), second)

    def test_nyc_row_context_parses_hour_date_and_airport_zone_ids(self):
        hour, day = _infer_hour_day({"hour_of_day": 23, "date": "2022-01-01"})

        self.assertEqual(hour, 23)
        self.assertEqual(day, 5)  # Saturday; datetime-derived weekdays are already zero based.
        self.assertTrue(_infer_airport_trip({"pickup_location": 132, "dropoff_location": 161}))
        self.assertFalse(_infer_airport_trip({"pickup_location": 170, "dropoff_location": 161}))

    def test_dataset_comparison_records_the_policy_action_and_tariff(self):
        core = Core(
            market_name="New York City",
            seed=13,
            choice_mode="parametric",
            firm1_mode="RL",
            firm2_mode="static",
            total_customers_pool=20,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "rides.csv")
            output = os.path.join(directory, "comparison.csv")
            model_path = os.path.join(directory, "trained-policy.pt")
            core.save_trained_model(
                model_path,
                metadata={"training_seed": 13},
                model_id="test-policy",
            )
            core.load_trained_model(model_path)
            with open(source, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "business", "pickup_location", "dropoff_location", "trip_length",
                    "request_to_dropoff", "date", "hour_of_day", "passenger_fare",
                ])
                writer.writeheader()
                writer.writerow({
                    "business": "Uber",
                    "pickup_location": 132,
                    "dropoff_location": 161,
                    "trip_length": 8.0,
                    "request_to_dropoff": 1800,
                    "date": "2022-01-01",
                    "hour_of_day": 23,
                    "passenger_fare": 42.0,
                })

            summary = core.compare_trained_rl_to_dataset(
                dataset_root=directory,
                dataset_glob="*.csv",
                out_csv=output,
                out_plot_prefix=None,
                max_rows=1,
                preview_rows=0,
                policy_mode="argmax",
                dataset_sample_seed=101,
                duration_mode="predicted_only",
            )
            with open(output, newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(summary["rows_compared"], 1)
        self.assertEqual(summary["policy_mode"], "argmax")
        self.assertIn("policy_action", row)
        self.assertIn("policy_action_label", row)
        self.assertIn("anchor_tariff_price", row)
        self.assertIn("predicted_base_fare", row)
        self.assertEqual(row["airport"], "True")
        self.assertEqual(row["hour"], "23")
        self.assertEqual(row["duration_mode"], "predicted_only")
        self.assertEqual(row["dataset_sample_seed"], "101")
        self.assertEqual(summary["duration_mode"], "predicted_only")
        self.assertEqual(summary["dataset_sample_seed"], 101)
        self.assertEqual(summary["trained_policy_model_id"], "test-policy")
        self.assertEqual(summary["trained_policy_training_seed"], 13)
        self.assertEqual(row["actual_paid_source_column"], "passenger_fare")
        self.assertEqual(row["distance_source_column"], "trip_length")
        self.assertEqual(row["actual_paid_excluded_from_policy_input"], "True")
        self.assertEqual(row["trained_policy_model_id"], "test-policy")
        self.assertEqual(row["trained_policy_training_seed"], "13")
        self.assertIn('"passenger_fare":"42.0"', row["raw_record_json"])
        self.assertNotIn("actual_paid", row["policy_input_json"])

    @unittest.skipIf(core_module.pq is None, "pyarrow is not installed")
    def test_seeded_parquet_sampling_is_reproducible_and_changes_partitions(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "rides.parquet")
            pq.write_table(
                pa.table({"row_id": list(range(60)), "price": [10.0] * 60}),
                source,
                row_group_size=5,
            )

            def sampled_ids(seed):
                ids = []
                for _, records in _iter_tabular_rows(source, sample_seed=seed):
                    ids.extend(int(row["row_id"]) for row in records)
                    if len(ids) >= 15:
                        break
                return ids[:15]

            first = sampled_ids(7)
            repeated = sampled_ids(7)
            different = sampled_ids(11)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)

    def test_saving_same_alias_twice_preserves_both_registry_archives(self):
        core = Core(
            market_name="New York City",
            seed=17,
            choice_mode="parametric",
            firm1_mode="RL",
            firm2_mode="static",
            total_customers_pool=20,
        )
        with tempfile.TemporaryDirectory() as directory:
            alias = os.path.join(directory, "latest.pt")
            registry = os.path.join(directory, "registry")

            first = core.save_trained_model(
                alias, registry_dir=registry, model_id="experiment-17"
            )
            second = core.save_trained_model(
                alias, registry_dir=registry, model_id="experiment-17"
            )

            self.assertNotEqual(first, second)
            self.assertTrue(os.path.isfile(first))
            self.assertTrue(os.path.isfile(second))
            self.assertTrue(os.path.isfile(alias))
            self.assertEqual(resolve_model_reference(os.path.basename(first), registry), first)
            loaded = core.load_trained_model(first)
            self.assertEqual(loaded["model_id"], "experiment-17")
            self.assertEqual(core.loaded_trained_model_path, first)


if __name__ == "__main__":
    unittest.main()
