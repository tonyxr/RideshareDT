import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import v28_threshold_batch as batch


class _FakePromptCore:
    def __init__(self, model=batch.DEFAULT_MODEL):
        self.model = model

    def _build_gpt_numeric_threshold_request_payload(self, pairs):
        prompt = {
            "city": "test",
            "evidence_protocol": {
                "prompt_protocol": "numeric_profile_threshold_v12_uncapped",
            },
            "profiles": [
                {"profile_index": index, "profile": profile, "rides": rides}
                for index, (profile, rides) in enumerate(pairs)
            ],
            "output_constraints": {
                "price_threshold_domain": "finite positive US dollars",
                "economic_bounds": "none",
                "expected_threshold_count": len(pairs),
            },
        }
        payload = {
            "model": self.model,
            "instructions": "Return one finite positive threshold per profile.",
            "input": json.dumps(prompt),
            "temperature": 0.0,
            "max_output_tokens": 500,
            "text": {"format": {"type": "json_schema", "name": "thresholds"}},
        }
        return payload, []


def _rows(count):
    rows = []
    for index in range(count):
        rows.append({
            "Age": 20 + index,
            "EmploymentStatus": "Employed",
            "Gender": "Female" if index % 2 else "Male",
            "MaritalStatus": "Single",
            "HouseholdSize": 1 + index,
            "IncomeBracket": "50k-100k",
            "LoyaltyFirm": "Firm1",
            "LoyaltyStrength": 0.51 + 0.01 * index,
            "LoyaltyType": "Returning",
            "ProfileMarket": "Chicago",
            "ColdstartRides": [
                {
                    "Hour": (index + ride) % 24,
                    "Weather": "clear",
                    "DistanceMiles": float(index + ride + 1),
                    "DurationMinutes": float(10 + index + ride),
                    "Service": "economy",
                    "Airport": bool((index + ride) % 2),
                    "DayOfWeek": (index + ride) % 7,
                }
                for ride in range(5)
            ],
            # Historical annotations must never leak into the native cache.
            "PriceThreshold": 5.0,
            "PriceThresholdSource": "legacy",
            "ThresholdPromptProtocol": "legacy_bounded",
        })
    return rows


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class V28ThresholdBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.jsonl"
        self.calibration = self.root / "calibration.json"
        self.calibration.write_text("{}\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _prepare(self, count=5, work_name="work", **overrides):
        source_rows = _rows(count)
        _write_jsonl(self.source, source_rows)
        kwargs = {
            "city": "chicago",
            "source_cache": self.source,
            "calibration_path": self.calibration,
            "work_dir": self.root / work_name,
            "expected_rows": count,
            "batch_size": 2,
            "model": batch.DEFAULT_MODEL,
            "coldstart_rides": 5,
            "max_shard_bytes": 10_000,
            "max_queued_tokens": 2_000,
            "chars_per_token": 2.5,
        }
        kwargs.update(overrides)
        with mock.patch.object(
            batch, "_build_prompt_core", return_value=_FakePromptCore()
        ):
            manifest = batch.prepare(**kwargs)
        return source_rows, Path(kwargs["work_dir"]), manifest

    @staticmethod
    def _output_for_shard(shard, *, omit_global=None, value_fn=None):
        mapping = json.loads(Path(shard["mapping_path"]).read_text(encoding="utf-8"))
        lines = []
        value_fn = value_fn or (lambda index: 1.0 + index)
        for request in mapping["requests"]:
            thresholds = []
            for profile in request["profiles"]:
                global_index = int(profile["global_index"])
                if global_index == omit_global:
                    continue
                thresholds.append({
                    "profile_index": int(profile["local_index"]),
                    "price_threshold": float(value_fn(global_index)),
                })
            body = {
                "id": f"resp_{request['custom_id']}",
                "model": batch.DEFAULT_MODEL,
                "output_text": json.dumps({"thresholds": thresholds}),
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
            }
            lines.append(json.dumps({
                "custom_id": request["custom_id"],
                "response": {"status_code": 200, "body": body},
                "error": None,
            }) + "\n")
        return "".join(lines).encode("utf-8")

    def _inject_all(self, work, manifest, *, omit_global=None, value_fn=None):
        for shard in manifest["shards"]:
            batch.inject_collected_output_for_test(
                work,
                shard_id=shard["shard_id"],
                payload=self._output_for_shard(
                    shard, omit_global=omit_global, value_fn=value_fn
                ),
            )

    def test_prepare_is_content_addressed_bounded_and_idempotent(self):
        _, work, manifest = self._prepare(count=7)
        self.assertEqual(manifest["endpoint"], "/v1/responses")
        self.assertEqual(manifest["model"], batch.DEFAULT_MODEL)
        self.assertEqual(
            manifest["prompt_protocol"],
            "numeric_profile_threshold_v12_uncapped",
        )
        self.assertEqual(manifest["estimate"]["profiles"], 7)
        self.assertEqual(manifest["estimate"]["requests"], 4)
        custom_ids = []
        for shard in manifest["shards"]:
            self.assertLess(shard["request_bytes"], 200_000_000)
            self.assertLessEqual(shard["estimated_prompt_tokens"], 2_000)
            self.assertEqual(
                batch.sha256_file(shard["request_path"]), shard["request_sha256"]
            )
            custom_ids.extend(shard["custom_ids"])
        self.assertEqual(len(custom_ids), len(set(custom_ids)))

        with mock.patch.object(
            batch, "_build_prompt_core", return_value=_FakePromptCore()
        ):
            second = batch.prepare(
                city="chicago",
                source_cache=self.source,
                calibration_path=self.calibration,
                work_dir=work,
                expected_rows=7,
                batch_size=2,
                model=batch.DEFAULT_MODEL,
                coldstart_rides=5,
                max_shard_bytes=10_000,
                max_queued_tokens=2_000,
                chars_per_token=2.5,
            )
        self.assertEqual(second["run_fingerprint"], manifest["run_fingerprint"])
        self.assertNotIn("OPENAI_API_KEY", (work / "manifest.json").read_text())

    def test_dry_run_writes_nothing_and_reports_cost(self):
        _write_jsonl(self.source, _rows(3))
        work = self.root / "estimate-only"
        with mock.patch.object(
            batch, "_build_prompt_core", return_value=_FakePromptCore()
        ):
            result = batch.prepare(
                city="chicago",
                source_cache=self.source,
                calibration_path=self.calibration,
                work_dir=work,
                expected_rows=3,
                batch_size=2,
                dry_run=True,
            )
        self.assertTrue(result["dry_run"])
        self.assertGreater(result["estimated_prompt_tokens"], 0)
        self.assertGreater(result["estimated_batch_cost_usd"], 0.0)
        self.assertGreaterEqual(
            result["batch_cost_at_output_ceiling_usd"],
            result["estimated_batch_cost_usd"],
        )
        self.assertFalse(work.exists())

    def test_validate_preserves_order_and_accepts_uncapped_values(self):
        source_rows, work, manifest = self._prepare(count=5)
        self._inject_all(
            work,
            manifest,
            value_fn=lambda index: 17.75 if index == 4 else 0.25 + index,
        )
        output = self.root / "native.jsonl"
        report = batch.validate(work, output_cache=output)
        self.assertEqual(report["rows"], 5)
        cached = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([row["Age"] for row in cached], [row["Age"] for row in source_rows])
        self.assertEqual(cached[4]["PriceThreshold"], 17.75)
        self.assertEqual(cached[4]["ThresholdRawLLMValue"], 17.75)
        self.assertEqual(cached[4]["PriceThresholdSource"], "llm_numeric")
        self.assertFalse(cached[4]["ThresholdLLMClipped"])
        self.assertTrue(cached[4]["ExperimentalUnboundedEconomics"])
        self.assertEqual(
            cached[4]["ThresholdEconomicContract"],
            "v28_unbounded_finite_positive",
        )
        self.assertEqual(cached[0]["ColdstartRides"], source_rows[0]["ColdstartRides"])
        self.assertEqual(cached[0]["Gender"], source_rows[0]["Gender"])
        self.assertNotEqual(cached[0]["ThresholdPromptProtocol"], "legacy_bounded")
        self.assertEqual(report["usage"]["input_tokens"], 300)

    def test_missing_row_blocks_cache_then_retry_contains_only_missing(self):
        _, work, manifest = self._prepare(count=5)
        self._inject_all(work, manifest, omit_global=4)
        with self.assertRaisesRegex(batch.ThresholdBatchError, "profiles lack valid"):
            batch.validate(work, output_cache=self.root / "blocked.jsonl")
        with mock.patch.object(
            batch, "_build_prompt_core", return_value=_FakePromptCore()
        ):
            retry = batch.retry_missing(work)
        self.assertEqual(retry["missing_profiles"], 1)
        _, retry_manifest = batch.load_manifest(work)
        retry_shards = [shard for shard in retry_manifest["shards"] if shard["attempt"] == 1]
        self.assertEqual(len(retry_shards), 1)
        mapping = json.loads(Path(retry_shards[0]["mapping_path"]).read_text())
        self.assertEqual(mapping["requests"][0]["global_indices"], [4])
        batch.inject_collected_output_for_test(
            work,
            shard_id=retry_shards[0]["shard_id"],
            payload=self._output_for_shard(
                retry_shards[0], value_fn=lambda _: 19.5
            ),
        )
        report = batch.validate(work, output_cache=self.root / "retried.jsonl")
        self.assertEqual(report["threshold_distribution"]["max"], 19.5)
        self.assertTrue(report["historical_parse_errors_resolved_by_retry"])

    def test_retry_refuses_to_duplicate_unsubmitted_original_shards(self):
        _, work, _ = self._prepare(count=2)
        with self.assertRaisesRegex(batch.ThresholdBatchError, "before every original"):
            batch.retry_missing(work)

    def test_submit_and_collect_are_explicit_and_never_persist_key(self):
        _, work, manifest = self._prepare(count=2)
        secret = "sk-unit-test-secret-do-not-persist"
        upload = {"id": "file_test"}
        submitted = {"id": "batch_test", "status": "validating"}
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": secret}), mock.patch.object(
            batch, "_upload_batch_file", return_value=upload
        ) as upload_mock, mock.patch.object(
            batch, "_api_json", return_value=submitted
        ) as api_mock:
            result = batch.submit(work)
        self.assertEqual(result["submitted_shards"], 1)
        upload_mock.assert_called_once()
        self.assertEqual(api_mock.call_args.args[:2], ("POST", "/batches"))
        self.assertNotIn(secret, (work / "manifest.json").read_text())

        shard = manifest["shards"][0]
        output_payload = self._output_for_shard(shard)
        complete = {
            "id": "batch_test",
            "status": "completed",
            "output_file_id": "file_output",
            "error_file_id": None,
            "request_counts": {"total": 1, "completed": 1, "failed": 0},
        }
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": secret}), mock.patch.object(
            batch, "_api_json", return_value=complete
        ), mock.patch.object(batch, "_api_bytes", return_value=output_payload):
            collected = batch.collect(work)
        self.assertEqual(collected["downloaded_files"], 1)
        self.assertNotIn(secret, (work / "manifest.json").read_text())

    def test_failed_batch_validation_errors_are_sanitized_and_persisted(self):
        _, work, _ = self._prepare(count=2)
        api_key = "sk-unit-test-api-key-abcdefghijklmnop"
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": api_key}), mock.patch.object(
            batch, "_upload_batch_file", return_value={"id": "file_test"}
        ), mock.patch.object(
            batch,
            "_api_json",
            return_value={"id": "batch_test", "status": "validating"},
        ):
            batch.submit(work)

        echoed_secret = "sk-proj-should-never-be-persisted-123456"
        failed = {
            "id": "batch_test",
            "status": "failed",
            "request_counts": {"total": 0, "completed": 0, "failed": 0},
            "errors": {
                "object": "list",
                "data": [
                    {
                        "code": "invalid_request",
                        "line": 17,
                        "message": f"bad payload Bearer {echoed_secret}",
                        "param": "body.model",
                        "arbitrary_nested_debug": {
                            "authorization": f"Bearer {echoed_secret}"
                        },
                    }
                ],
            },
        }
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": api_key}), mock.patch.object(
            batch, "_api_json", return_value=failed
        ):
            result = batch.status(work)
        self.assertEqual(result["statuses"], {"failed": 1})
        manifest_text = (work / "manifest.json").read_text(encoding="utf-8")
        self.assertNotIn(api_key, manifest_text)
        self.assertNotIn(echoed_secret, manifest_text)
        _, manifest = batch.load_manifest(work)
        errors = manifest["shards"][0]["batch"]["errors"]
        self.assertEqual(errors["source_error_count"], 1)
        self.assertEqual(errors["data"][0]["code"], "invalid_request")
        self.assertEqual(errors["data"][0]["line"], 17)
        self.assertIn("[REDACTED]", errors["data"][0]["message"])
        self.assertNotIn("arbitrary_nested_debug", errors["data"][0])

    def test_requeue_zero_request_failure_archives_attempt_and_uses_new_generation(self):
        _, work, prepared = self._prepare(count=2)
        shard_id = prepared["shards"][0]["shard_id"]
        api_key = "sk-unit-test-api-key-abcdefghijklmnop"
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": api_key}), mock.patch.object(
            batch, "_upload_batch_file", return_value={"id": "file_test"}
        ), mock.patch.object(
            batch,
            "_api_json",
            return_value={"id": "batch_failed", "status": "validating"},
        ):
            batch.submit(work)
        failed = {
            "id": "batch_failed",
            "status": "failed",
            "request_counts": {"total": 0, "completed": 0, "failed": 0},
            "errors": {
                "object": "list",
                "data": [{"code": "token_limit_exceeded", "message": "queue full"}],
            },
        }
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": api_key}), mock.patch.object(
            batch, "_api_json", return_value=failed
        ):
            batch.status(work)

        transitioned = batch.requeue_failed_validation(
            work, shard_ids=[shard_id]
        )
        self.assertEqual(transitioned["requeued_shards"], [shard_id])
        _, manifest = batch.load_manifest(work)
        shard = manifest["shards"][0]
        self.assertEqual(shard["submission_generation"], 1)
        self.assertIsNone(shard["batch_id"])
        self.assertIsNone(shard["batch"])
        self.assertEqual(shard["input_file_id"], "file_test")
        self.assertEqual(len(shard["submission_history"]), 1)
        archived = shard["submission_history"][0]
        self.assertEqual(archived["batch_id"], "batch_failed")
        self.assertEqual(
            archived["batch"]["errors"]["data"][0]["code"],
            "token_limit_exceeded",
        )
        batch.verify_manifest_files(manifest)
        archived_copy = json.loads(json.dumps(archived, sort_keys=True))

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": api_key}), mock.patch.object(
            batch, "_upload_batch_file"
        ) as upload_mock, mock.patch.object(
            batch,
            "_api_json",
            return_value={"id": "batch_retry", "status": "validating"},
        ) as api_mock:
            batch.submit(work)
        upload_mock.assert_not_called()
        self.assertTrue(
            api_mock.call_args.kwargs["extra_headers"]["Idempotency-Key"].endswith(
                "-g1"
            )
        )
        _, retried = batch.load_manifest(work)
        self.assertEqual(
            retried["shards"][0]["submission_history"][0], archived_copy
        )
        self.assertEqual(retried["shards"][0]["batch_id"], "batch_retry")

    def test_requeue_refuses_any_partially_processed_failure(self):
        _, work, prepared = self._prepare(count=2)
        shard_id = prepared["shards"][0]["shard_id"]
        api_key = "sk-unit-test-api-key-abcdefghijklmnop"
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": api_key}), mock.patch.object(
            batch, "_upload_batch_file", return_value={"id": "file_test"}
        ), mock.patch.object(
            batch,
            "_api_json",
            return_value={"id": "batch_partial", "status": "validating"},
        ):
            batch.submit(work)
        partial = {
            "id": "batch_partial",
            "status": "failed",
            "request_counts": {"total": 1, "completed": 0, "failed": 1},
            "errors": {"object": "list", "data": []},
        }
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": api_key}), mock.patch.object(
            batch, "_api_json", return_value=partial
        ):
            batch.status(work)
        with self.assertRaisesRegex(batch.ThresholdBatchError, "not safe to requeue"):
            batch.requeue_failed_validation(work, shard_ids=[shard_id])
        _, manifest = batch.load_manifest(work)
        self.assertEqual(manifest["shards"][0]["batch_id"], "batch_partial")
        self.assertEqual(manifest["shards"][0].get("submission_history", []), [])

    def test_cross_manifest_queue_guard_blocks_concurrent_city_submission(self):
        _, first_work, first = self._prepare(count=2, work_name="first")
        _, second_work, _ = self._prepare(count=2, work_name="second")
        api_key = "sk-unit-test-api-key-abcdefghijklmnop"
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": api_key}), mock.patch.object(
            batch, "_upload_batch_file", return_value={"id": "file_first"}
        ), mock.patch.object(
            batch,
            "_api_json",
            return_value={"id": "batch_active", "status": "in_progress"},
        ):
            batch.submit(first_work)
        active_tokens = int(first["shards"][0]["estimated_prompt_tokens"])
        ledger = batch.queue_ledger(
            [first_work, second_work],
            global_max_active_estimated_tokens=active_tokens,
        )
        self.assertEqual(ledger["active_estimated_tokens"], active_tokens)
        self.assertEqual(ledger["available_estimated_tokens"], 0)
        self.assertEqual(ledger["prepared_jobs_fitting_now"], [])
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": api_key}), mock.patch.object(
            batch, "_upload_batch_file"
        ) as upload_mock:
            with self.assertRaisesRegex(batch.ThresholdBatchError, "global queued-token"):
                batch.submit(
                    second_work,
                    peer_work_dirs=[first_work],
                    global_max_active_estimated_tokens=active_tokens,
                )
        upload_mock.assert_not_called()

    def test_source_validation_rejects_duplicates(self):
        rows = _rows(2)
        rows[1] = json.loads(json.dumps(rows[0]))
        _write_jsonl(self.source, rows)
        with self.assertRaisesRegex(batch.ThresholdBatchError, "not unique"):
            batch.load_source_rows(
                self.source,
                market_name="Chicago",
                expected_rows=2,
                coldstart_rides=5,
            )


if __name__ == "__main__":
    unittest.main()
