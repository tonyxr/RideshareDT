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
