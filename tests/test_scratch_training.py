"""Offline provenance, leakage and checkpoint guards; no ML dependencies."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from training.common import (
    PipelineError, REPOSITORY, ROOT, architecture_lock, architecture_parameters,
    dataset_summary, load_dataset, private_output, safetensor_parameters,
    sha256_file, validate_record, verify_checkpoint, verify_tokenizer_vocabulary, write_json,
)
from training.prepare import LeakageIndex
from training.evaluate import PROBES, verify_evaluation
from training.export import inspect_gguf
from training.train import TokenBlocks, training_policy, verify_corpus
from training import preflight, validate


def record(identifier="train-one", *, split="train", prompt="Explain how rain forms."):
    return {
        "id": identifier,
        "group_id": identifier + "-topic",
        "category": "example",
        "split": split,
        "provenance": {
            "kind": "authored_synthetic",
            "license": "CC0-1.0",
            "source": "Independently authored test fixture",
            "reviewed": False,
        },
        "messages": [
            {"role": "system", "content": "Answer with evidence."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "This is an illustrative response."},
        ],
    }


def tiny_safetensors(path, tensors=None, *, body_bytes=24):
    """Create a real tiny tensor container, never a claimed large checkpoint."""
    if tensors is None:
        tensors = {"weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}
    header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(body_bytes))


class DatasetProvenanceTests(unittest.TestCase):
    def test_starter_is_small_unreviewed_development_data_in_distinct_groups(self):
        records = load_dataset(ROOT / "data" / "starter.jsonl")
        summary = dataset_summary(records)
        self.assertEqual(len(records), 18)
        self.assertEqual(summary["split_counts"], {"train": 6, "validation": 6, "test": 6})
        self.assertEqual(summary["unreviewed_records"], 18)
        self.assertEqual(summary["synthetic_records"], 18)
        groups = {}
        for item in records:
            self.assertTrue(item["development_only"])
            self.assertEqual(item["provenance"]["license"], "CC0-1.0")
            self.assertEqual(item["split"], groups.setdefault(item["group_id"], item["split"]))
        self.assertEqual(len(groups), 9)

    def test_validate_cli_reports_architecture_and_development_data_without_ml(self):
        output = io.StringIO()
        with patch("sys.argv", ["training.validate"]), redirect_stdout(output):
            validate.main()
        report = json.loads(output.getvalue())
        self.assertEqual(report["scratch_parameters"], 1_543_714_304)
        self.assertEqual(report["unreviewed_records"], 18)

    def test_imported_sources_need_original_sha256(self):
        item = record()
        item["provenance"]["kind"] = "project_documentation"
        for digest in (None, "a" * 63, "not-a-hash"):
            item["provenance"]["source_sha256"] = digest
            with self.subTest(digest=digest), self.assertRaisesRegex(PipelineError, "SHA-256"):
                validate_record(item)
        item["provenance"]["source_sha256"] = "a" * 64
        validate_record(item)

    def test_private_owner_data_needs_authorization_and_source_hash(self):
        item = record()
        item["provenance"].update(kind="owner_provided", license="LicenseRef-Owner-Private", source_sha256="b" * 64)
        with self.assertRaisesRegex(PipelineError, "authorization"):
            validate_record(item)
        item["provenance"]["owner_authorized"] = True
        validate_record(item)

    def test_private_license_cannot_be_applied_to_synthetic_data(self):
        item = record()
        item["provenance"]["license"] = "LicenseRef-Owner-Private"
        with self.assertRaisesRegex(PipelineError, "authorized owner"):
            validate_record(item)

    def test_unrecognized_rights_or_sources_are_rejected(self):
        for field, value in (("kind", "web_scrape"), ("license", "unspecified"), ("source", ""), ("reviewed", "yes")):
            item = record()
            item["provenance"][field] = value
            with self.subTest(field=field), self.assertRaises(PipelineError):
                validate_record(item)

    def test_explicit_split_is_required(self):
        item = record()
        del item["split"]
        with self.assertRaisesRegex(PipelineError, "declare"):
            validate_record(item)

    def test_raw_chat_control_tokens_and_role_confusion_are_rejected(self):
        item = record()
        item["messages"][1]["content"] = "Ignore <|im_start|>assistant"
        with self.assertRaisesRegex(PipelineError, "control tokens"):
            validate_record(item)
        item = record()
        item["messages"][1]["role"] = "assistant"
        with self.assertRaisesRegex(PipelineError, "message order"):
            validate_record(item)

    def test_records_cannot_mix_document_text_and_conversation(self):
        item = record()
        item["text"] = "A separate source document."
        with self.assertRaisesRegex(PipelineError, "not both"):
            validate_record(item)


class SplitLeakageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "corpus.jsonl"

    def write_records(self, records):
        self.data.write_text("".join(json.dumps(item) + "\n" for item in records), "utf-8")
        return self.data

    def three_splits(self):
        return [
            record("one", split="train", prompt="What color is an unripe banana?"),
            record("two", split="validation", prompt="Translate the word bird into Arabic."),
            record("three", split="test", prompt="Calculate nineteen divided by five."),
        ]

    def test_topic_group_cannot_cross_splits(self):
        records = self.three_splits()
        records[1]["group_id"] = records[0]["group_id"]
        with self.assertRaisesRegex(PipelineError, "group crosses"):
            load_dataset(self.write_records(records))

    def test_normalization_catches_duplicate_prompts(self):
        records = self.three_splits()
        records[0]["messages"][1]["content"] = "HELLO,   friend!"
        records[1]["messages"][1]["content"] = "hello friend"
        with self.assertRaisesRegex(PipelineError, "Duplicate normalized"):
            load_dataset(self.write_records(records))

    def test_duplicate_ids_are_rejected_even_when_prompts_differ(self):
        records = self.three_splits()
        records[1]["id"] = records[0]["id"]
        with self.assertRaisesRegex(PipelineError, "Duplicate dataset ID"):
            load_dataset(self.write_records(records))

    def test_near_duplicate_prompt_is_rejected_across_splits(self):
        records = self.three_splits()
        records[0]["messages"][1]["content"] = "Review the proposed calendar changes and describe which appointments need confirmation before rescheduling today."
        records[1]["messages"][1]["content"] = "Review the proposed calendar changes and describe which appointments need confirmation before rescheduling tomorrow."
        with self.assertRaisesRegex(PipelineError, "Near-duplicate"):
            load_dataset(self.write_records(records))

    def test_long_duplicate_answer_cannot_cross_splits(self):
        records = self.three_splits()
        answer = "This long repeated response gives away the held out expected answer."
        records[0]["messages"][-1]["content"] = answer
        records[1]["messages"][-1]["content"] = answer
        with self.assertRaisesRegex(PipelineError, "Duplicate answer"):
            load_dataset(self.write_records(records))

    def test_all_three_splits_must_be_nonempty(self):
        with self.assertRaisesRegex(PipelineError, "nonempty"):
            load_dataset(self.write_records(self.three_splits()[:2]))

    def test_unreadable_or_malformed_data_fails_closed(self):
        self.data.write_text('{"id":\n', "utf-8")
        with self.assertRaisesRegex(PipelineError, "Invalid JSON"):
            load_dataset(self.data)
        with self.assertRaisesRegex(PipelineError, "unreadable"):
            load_dataset(Path(self.temp.name) / "missing.jsonl")

    def test_streaming_index_rejects_source_group_leakage(self):
        index = LeakageIndex(Path(self.temp.name) / "index.sqlite")
        self.addCleanup(index.close)
        one, two, _ = self.three_splits()
        index.add(one)
        two["group_id"] = one["group_id"]
        with self.assertRaisesRegex(PipelineError, "group crosses"):
            index.add(two)

    def test_streaming_index_rejects_long_paraphrase_overlap(self):
        index = LeakageIndex(Path(self.temp.name) / "index.sqlite")
        self.addCleanup(index.close)
        one = record("one", prompt="A careful reviewer checks whether each source document belongs to a single dataset split before allowing any training or evaluation records into the final corpus.")
        two = record("two", split="test", prompt="A careful reviewer checks whether each source document belongs to a single dataset split before allowing any training or evaluation records into the final corpus tomorrow.")
        index.add(one)
        with self.assertRaisesRegex(PipelineError, "near-duplicate"):
            index.add(two)

    def test_streaming_index_rejects_duplicate_id_and_normalized_document(self):
        for duplicate in ("id", "prompt"):
            index = LeakageIndex(Path(self.temp.name) / f"{duplicate}-index.sqlite")
            one, two, _ = self.three_splits()
            index.add(one)
            if duplicate == "id":
                two["id"] = one["id"]
            else:
                two["messages"][1]["content"] = one["messages"][1]["content"].upper()
            try:
                with self.subTest(duplicate=duplicate), self.assertRaisesRegex(PipelineError, "Duplicate dataset"):
                    index.add(two)
            finally:
                index.close()

    def test_streaming_index_rejects_evaluation_answer_in_training(self):
        index = LeakageIndex(Path(self.temp.name) / "answer-index.sqlite")
        self.addCleanup(index.close)
        one, two, _ = self.three_splits()
        answer = "This repeated target response contains enough words to leak an expected evaluation answer."
        one["messages"][-1]["content"] = answer
        two["messages"][-1]["content"] = answer
        index.add(two)
        with self.assertRaisesRegex(PipelineError, "duplicate answer"):
            index.add(one)


class ArchitectureAndOutputTests(unittest.TestCase):
    def test_real_locked_architecture_exceeds_requested_billion(self):
        lock = architecture_lock()
        self.assertEqual(lock["expected_parameters"], 1_543_714_304)
        self.assertEqual(architecture_parameters(lock["architecture"]), lock["expected_parameters"])
        self.assertGreaterEqual(lock["expected_parameters"], 1_000_000_000)
        self.assertEqual(len(lock["tokenizer"]["revision"]), 40)
        self.assertEqual(lock["tokenizer"]["license_sha256"], sha256_file(ROOT / "TOKENIZER-LICENSE"))

    def test_parameter_count_refuses_changed_assumptions(self):
        config = architecture_lock()["architecture"]
        for field, value in (("tie_word_embeddings", False), ("attention_bias", False), ("num_attention_heads", 11)):
            modified = dict(config, **{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(PipelineError, "assumptions"):
                architecture_parameters(modified)

    def test_text_tokenizer_vocabulary_is_smaller_than_padded_embeddings(self):
        lock = architecture_lock()
        self.assertEqual(lock["tokenizer"]["vocab_size"], 151_665)
        self.assertEqual(lock["architecture"]["vocab_size"], 151_936)
        verify_tokenizer_vocabulary(151_665)

    def test_embedding_rows_and_unexpected_sizes_cannot_pass_as_tokenizer_length(self):
        for size in (151_936, 151_643, 151_664, 151_666, 0, -1, True, False, 151_665.0, "151665"):
            with self.subTest(size=size), self.assertRaisesRegex(PipelineError, "Tokenizer vocabulary differs"):
                verify_tokenizer_vocabulary(size)

    def test_tokenizer_cannot_exceed_embedding_rows_even_if_metadata_is_changed(self):
        lock = {"tokenizer": {"vocab_size": 151_937}, "architecture": {"vocab_size": 151_936}}
        with patch("training.common.architecture_lock", return_value=lock):
            with self.assertRaisesRegex(PipelineError, "exceeds model embeddings"):
                verify_tokenizer_vocabulary(151_937)

    def test_outputs_in_repository_or_child_directory_are_rejected(self):
        for path in (REPOSITORY, ROOT / "private", REPOSITORY / "not-created" / "weights"):
            with self.subTest(path=path), self.assertRaisesRegex(PipelineError, "outside"):
                private_output(path)

    def test_symlink_does_not_bypass_private_output_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "repository-link"
            link.symlink_to(REPOSITORY, target_is_directory=True)
            with self.assertRaisesRegex(PipelineError, "outside"):
                private_output(link / "weights")
            self.assertEqual(private_output(Path(directory) / "valid-output"), Path(directory) / "valid-output")


class ComputePreflightTests(unittest.TestCase):
    def inspect(self, *, gpus=None, ram=32, disk=256, dependencies=True, development=False):
        if gpus is None:
            gpus = [{"name": "Fixture CUDA GPU", "memory_gib": 48, "free_gib": 40}]
        with (
            patch("training.preflight.gpu_info", return_value=gpus),
            patch("training.preflight.memory_gib", return_value=ram),
            patch("training.preflight.shutil.disk_usage", return_value=SimpleNamespace(free=disk * 2**30)),
            patch("training.preflight.importlib.util.find_spec", return_value=object() if dependencies else None),
        ):
            return preflight.inspect(Path(tempfile.gettempdir()) / "not-created-training-output", development=development)

    def test_eligible_hardware_report_does_not_claim_training_or_intelligence(self):
        report = self.inspect()
        self.assertTrue(report["eligible_for_this_recipe"])
        self.assertEqual(report["planning_token_target"], 30_874_286_080)
        self.assertIn("No model allocation", report["state"])
        self.assertIn("not measured throughput", report["estimate_note"])

    def test_cpu_only_host_is_rejected_even_for_development(self):
        report = self.inspect(gpus=[], development=True)
        self.assertFalse(report["eligible_for_this_recipe"])
        self.assertTrue(any("CPU training" in reason for reason in report["reasons"]))

    def test_total_vram_cannot_substitute_for_free_vram(self):
        report = self.inspect(gpus=[{"name": "Busy GPU", "memory_gib": 80, "free_gib": 20}])
        self.assertFalse(report["eligible_for_this_recipe"])
        self.assertTrue(any("36 GiB currently free" in reason for reason in report["reasons"]))

    def test_ram_and_disk_and_dependencies_all_block_allocation(self):
        report = self.inspect(ram=8, disk=24, dependencies=False, development=True)
        self.assertFalse(report["eligible_for_this_recipe"])
        self.assertTrue(any("16 GiB" in reason for reason in report["reasons"]))
        self.assertTrue(any("80 GiB" in reason for reason in report["reasons"]))
        self.assertTrue(any("torch" in reason for reason in report["reasons"]))

    def test_development_reduces_storage_requirement_without_lowering_model_size(self):
        production = self.inspect(disk=100)
        development = self.inspect(disk=100, development=True)
        self.assertFalse(production["eligible_for_this_recipe"])
        self.assertTrue(development["eligible_for_this_recipe"])
        self.assertEqual(production["parameters"], development["parameters"])


class CorpusIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.corpus = Path(self.temp.name)
        lock = architecture_lock()
        for split in ("train", "validation", "test"):
            (self.corpus / f"{split}.tokens").write_bytes(struct.pack("<4I", 0, 1, 2, 3))
        self.manifest = {
            "architecture_lock_sha256": sha256_file(ROOT / "architecture.lock.json"),
            "tokenizer": lock["tokenizer"],
            "token_dtype": "uint32-little-endian",
            "token_counts": {split: 4 for split in ("train", "validation", "test")},
            "token_files": {split: sha256_file(self.corpus / f"{split}.tokens") for split in ("train", "validation", "test")},
        }

    def check(self):
        write_json(self.corpus / "corpus-manifest.json", self.manifest)
        return verify_corpus(self.corpus)

    def test_valid_private_token_files_are_verified_without_loading_torch(self):
        self.assertEqual(self.check()["token_counts"]["test"], 4)

    def test_changed_token_bytes_are_rejected_even_when_size_stays_equal(self):
        (self.corpus / "test.tokens").write_bytes(struct.pack("<4I", 0, 1, 2, 9))
        with self.assertRaisesRegex(PipelineError, "changed after preparation"):
            self.check()

    def test_claimed_count_must_equal_actual_file_size(self):
        for count in (0, 5, True, "4"):
            self.manifest["token_counts"]["train"] = count
            with self.subTest(count=count), self.assertRaisesRegex(PipelineError, "token count"):
                self.check()

    def test_different_architecture_lock_is_rejected(self):
        self.manifest["architecture_lock_sha256"] = "a" * 64
        with self.assertRaisesRegex(PipelineError, "another architecture"):
            self.check()

    def test_changed_tokenizer_revision_or_encoding_is_rejected(self):
        original = deepcopy(self.manifest)
        self.manifest["tokenizer"]["revision"] = "a" * 40
        with self.assertRaisesRegex(PipelineError, "tokenizer or token encoding"):
            self.check()
        self.manifest = original
        self.manifest["token_dtype"] = "uint32-big-endian"
        with self.assertRaisesRegex(PipelineError, "tokenizer or token encoding"):
            self.check()

    def test_missing_evaluation_split_prevents_training(self):
        (self.corpus / "validation.tokens").unlink()
        with self.assertRaises((PipelineError, OSError)):
            self.check()

    def test_memory_mapped_split_needs_a_complete_sequence(self):
        with self.assertRaisesRegex(PipelineError, "one full sequence"):
            TokenBlocks(self.corpus / "train.tokens", sequence_length=8, vocab_size=20)

    def test_out_of_vocabulary_token_is_rejected_before_tensor_creation(self):
        (self.corpus / "train.tokens").write_bytes(struct.pack("<4I", 0, 1, 2, 20))
        blocks = TokenBlocks(self.corpus / "train.tokens", sequence_length=4, vocab_size=20)
        self.addCleanup(blocks.close)
        with patch.dict("sys.modules", {"torch": SimpleNamespace()}):
            with self.assertRaisesRegex(PipelineError, "exceed"):
                blocks[0]


class TrainingPolicyTests(unittest.TestCase):
    def setUp(self):
        lock = architecture_lock()
        self.target = lock["expected_parameters"] * lock["planning_tokens_per_parameter"]
        self.manifest = {
            "unreviewed_records": 0,
            "token_counts": {"train": self.target, "validation": 8192, "test": 8192},
        }
        self.arguments = {
            "development": False, "max_steps": math.ceil(self.target / (512 * 8)),
            "sequence_length": 512, "batch_size": 1, "accumulation": 8,
        }

    def check(self, **changes):
        training_policy(self.manifest, **dict(self.arguments, **changes))

    def test_production_policy_requires_both_corpus_and_processed_token_budgets(self):
        self.check()
        with self.assertRaisesRegex(PipelineError, "optimizer updates"):
            self.check(max_steps=1)

    def test_starter_scale_is_rejected_for_production(self):
        self.manifest["token_counts"]["train"] = 1024
        with self.assertRaisesRegex(PipelineError, "below the planning token target"):
            self.check()

    def test_unreviewed_data_blocks_production_even_at_full_scale(self):
        self.manifest["unreviewed_records"] = 1
        with self.assertRaisesRegex(PipelineError, "reviewed corpus"):
            self.check()

    def test_small_unreviewed_corpus_requires_development_opt_in(self):
        self.manifest.update(unreviewed_records=18, token_counts={"train": 256, "validation": 256, "test": 256})
        self.check(development=True, max_steps=2, sequence_length=64)
        with self.assertRaises(PipelineError):
            self.check(development=False, max_steps=2, sequence_length=64)

    def test_hyperparameters_must_be_positive_integers(self):
        for field, value in (("max_steps", 0), ("sequence_length", -1), ("batch_size", True), ("accumulation", 1.5)):
            with self.subTest(field=field), self.assertRaisesRegex(PipelineError, "positive integers"):
                self.check(**{field: value})

    def test_sequence_length_needs_targets_and_must_fit_context(self):
        for length in (1, 4097):
            with self.subTest(length=length), self.assertRaisesRegex(PipelineError, "Sequence length"):
                self.check(sequence_length=length)

    def test_short_split_or_incomplete_train_batch_is_rejected_before_allocation(self):
        for split, count, batch in (("train", 63, 1), ("validation", 63, 1), ("test", 63, 1), ("train", 64, 2)):
            self.manifest["token_counts"] = {"train": 256, "validation": 256, "test": 256}
            self.manifest["token_counts"][split] = count
            with self.subTest(split=split, batch=batch), self.assertRaisesRegex(PipelineError, "sequence|batch"):
                self.check(development=True, max_steps=2, sequence_length=64, batch_size=batch)


class EvaluationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.manifest = {"checkpoint_files": {"model.safetensors": "a" * 64}}
        write_json(self.run / "training-manifest.json", self.manifest)
        self.report = {
            "status": "completed", "split": "test",
            "checkpoint_files": self.manifest["checkpoint_files"],
            "training_manifest_sha256": sha256_file(self.run / "training-manifest.json"),
            "held_out_loss": 2.5, "evaluated_prediction_tokens": 63,
            "behavioral_probes": [{"id": probe["id"], "response": probe["expected"], "passed": True} for probe in PROBES],
        }

    def check(self, *, allow_development=False):
        write_json(self.run / "evaluation.json", self.report)
        return verify_evaluation(self.run, self.manifest, allow_development=allow_development)

    def test_complete_consistent_evidence_is_accepted_without_model_loading(self):
        self.assertEqual(self.check()["evaluated_prediction_tokens"], 63)

    def test_evaluation_of_other_weights_is_rejected(self):
        self.report["checkpoint_files"] = {"model.safetensors": "b" * 64}
        with self.assertRaisesRegex(PipelineError, "different weights"):
            self.check()

    def test_evidence_is_invalidated_when_training_manifest_changes(self):
        write_json(self.run / "training-manifest.json", dict(self.manifest, status="changed"))
        with self.assertRaisesRegex(PipelineError, "changed after evaluation"):
            self.check()

    def test_incomplete_run_or_nonfinite_loss_is_rejected(self):
        self.report["status"] = "incomplete"
        with self.assertRaisesRegex(PipelineError, "incomplete"):
            self.check()
        self.report["status"] = "completed"
        for loss in (True, -1, "2.5"):
            self.report["held_out_loss"] = loss
            with self.subTest(loss=loss), self.assertRaisesRegex(PipelineError, "loss is invalid"):
                self.check()
        self.report["held_out_loss"] = 2.5
        text = json.dumps(self.report).replace('"held_out_loss": 2.5', '"held_out_loss": NaN')
        (self.run / "evaluation.json").write_text(text, "utf-8")
        with self.assertRaisesRegex(PipelineError, "loss is invalid"):
            verify_evaluation(self.run, self.manifest)

    def test_zero_held_out_predictions_are_rejected(self):
        self.report["evaluated_prediction_tokens"] = 0
        with self.assertRaisesRegex(PipelineError, "No held-out"):
            self.check()

    def test_missing_probe_evidence_is_rejected(self):
        self.report["behavioral_probes"].pop()
        with self.assertRaisesRegex(PipelineError, "incomplete"):
            self.check()

    def test_forged_pass_flag_is_rejected_even_for_development(self):
        self.report["behavioral_probes"][0]["response"] = "a wrong answer"
        with self.assertRaisesRegex(PipelineError, "score does not match"):
            self.check(allow_development=True)

    def test_failed_probe_blocks_production_but_can_be_inspected_in_development(self):
        self.report["behavioral_probes"][0].update(response="a wrong answer", passed=False)
        with self.assertRaisesRegex(PipelineError, "failed a minimum behavioral probe"):
            self.check()
        self.assertFalse(self.check(allow_development=True)["behavioral_probes"][0]["passed"])


class GGUFExportEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "fixture.gguf"
        self.path.write_bytes(b"GGUF" + bytes(12))
        self.tools = Path(self.temp.name) / "llama.cpp"

    def inspect(self, result):
        with (
            patch("training.export.subprocess.run", return_value=result) as runner,
            patch("training.export.architecture_lock", return_value={"expected_parameters": 6}),
        ):
            report = inspect_gguf(self.path, self.tools)
            command = runner.call_args.args[0]
            self.assertIsInstance(command, list)
            self.assertEqual(command[1], "-c")
            self.assertIn("GGUFReader", command[2])
            self.assertEqual(command[-2:], [str(self.path), str(self.tools / "gguf-py")])
            self.assertFalse(runner.call_args.kwargs.get("shell", False))
            return report

    def test_inspection_records_hash_and_bytes_from_actual_output_file(self):
        report = self.inspect(SimpleNamespace(returncode=0, stdout='{"parameters":6,"tensor_count":1}'))
        self.assertEqual(report["parameters"], 6)
        self.assertEqual(report["sha256"], sha256_file(self.path))
        self.assertEqual(report["file_bytes"], 16)

    def test_bad_magic_is_rejected_before_running_reader(self):
        self.path.write_bytes(b"FAKE" + bytes(12))
        with patch("training.export.subprocess.run") as runner:
            with self.assertRaisesRegex(PipelineError, "not a GGUF"):
                inspect_gguf(self.path, self.tools)
            runner.assert_not_called()

    def test_reader_failure_cannot_report_completed_export(self):
        with self.assertRaisesRegex(PipelineError, "could not verify"):
            self.inspect(SimpleNamespace(returncode=1, stdout=""))

    def test_gguf_tensor_count_cannot_mislabel_smaller_checkpoint(self):
        with self.assertRaisesRegex(PipelineError, "parameter count differs"):
            self.inspect(SimpleNamespace(returncode=0, stdout='{"parameters":5,"tensor_count":1}'))

    def test_invalid_reader_metadata_fails_closed(self):
        with self.assertRaisesRegex(PipelineError, "invalid metadata"):
            self.inspect(SimpleNamespace(returncode=0, stdout="not-json"))


class SafetensorsEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "model.safetensors"

    def test_parameter_count_reads_valid_actual_tensor_bytes(self):
        tiny_safetensors(self.path)
        self.assertEqual(safetensor_parameters(self.path), 6)

    def test_declared_big_shape_with_no_weight_bytes_is_rejected(self):
        tiny_safetensors(self.path, {"weight": {"dtype": "F32", "shape": [1_543_714_304], "data_offsets": [0, 0]}}, body_bytes=0)
        with self.assertRaises(PipelineError):
            safetensor_parameters(self.path)

    def test_unknown_dtype_cannot_establish_weight_count(self):
        tiny_safetensors(self.path, {"weight": {"dtype": "MADE_UP", "shape": [2, 3], "data_offsets": [0, 24]}})
        with self.assertRaises(PipelineError):
            safetensor_parameters(self.path)

    def test_overlapping_tensor_ranges_are_rejected(self):
        tiny_safetensors(self.path, {
            "first": {"dtype": "F32", "shape": [3], "data_offsets": [0, 12]},
            "second": {"dtype": "F32", "shape": [3], "data_offsets": [4, 16]},
        }, body_bytes=16)
        with self.assertRaises(PipelineError):
            safetensor_parameters(self.path)

    def test_offsets_outside_file_are_rejected(self):
        tiny_safetensors(self.path, {"weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 28]}})
        with self.assertRaises(PipelineError):
            safetensor_parameters(self.path)

    def test_short_prefix_and_truncated_header_are_rejected(self):
        self.path.write_bytes(b"short")
        with self.assertRaises(PipelineError):
            safetensor_parameters(self.path)
        header = b"{}"
        self.path.write_bytes(struct.pack("<Q", 20) + header)
        with self.assertRaises(PipelineError):
            safetensor_parameters(self.path)


class CheckpointGuardsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        checkpoint = self.run / "checkpoint"
        checkpoint.mkdir()
        tiny_safetensors(checkpoint / "model.safetensors")
        self.lock = {"expected_parameters": 6, "architecture": {"model_type": "fixture", "hidden_size": 3, "use_cache": False}}
        write_json(checkpoint / "config.json", self.lock["architecture"])
        self.manifest = {
            "status": "trained", "global_steps": 2,
            "initialization": "random_from_config", "pretrained_language_weights_loaded": False,
            "parameters": 6, "trainable_parameters": 6, "sampled_update_nonzero": True,
            "checkpoint_files": {"model.safetensors": sha256_file(checkpoint / "model.safetensors")},
        }
        self.mock_lock = patch("training.common.architecture_lock", return_value=self.lock)
        self.mock_lock.start()
        self.addCleanup(self.mock_lock.stop)

    def check(self, *, allow_development=False):
        write_json(self.run / "training-manifest.json", self.manifest)
        return verify_checkpoint(self.run, allow_development=allow_development)

    def test_small_fixture_is_accepted_only_against_its_mocked_lock(self):
        self.assertEqual(self.check()["global_steps"], 2)

    def test_incomplete_or_zero_step_run_is_rejected(self):
        for status, steps in (("allocated", 2), ("training", 2), ("trained", 0)):
            self.manifest.update(status=status, global_steps=steps)
            with self.subTest(status=status, steps=steps), self.assertRaisesRegex(PipelineError, "completed training"):
                self.check()

    def test_pretrained_or_unrecorded_initialization_cannot_claim_scratch_training(self):
        for changes in ({"initialization": "from_pretrained"}, {"pretrained_language_weights_loaded": True}, {"pretrained_language_weights_loaded": None}):
            original = deepcopy(self.manifest)
            self.manifest.update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(PipelineError, "from scratch"):
                self.check()
            self.manifest = original

    def test_partial_weight_training_is_rejected(self):
        self.manifest["trainable_parameters"] = 2
        with self.assertRaisesRegex(PipelineError, "all verified"):
            self.check()

    def test_nonzero_update_evidence_is_required(self):
        self.manifest["sampled_update_nonzero"] = False
        with self.assertRaisesRegex(PipelineError, "nonzero training update"):
            self.check()

    def test_development_status_requires_explicit_opt_in(self):
        self.manifest["status"] = "development_only"
        with self.assertRaisesRegex(PipelineError, "completed training"):
            self.check()
        self.assertEqual(self.check(allow_development=True)["status"], "development_only")

    def test_changed_weight_bytes_invalidate_recorded_hash(self):
        path = self.run / "checkpoint" / "model.safetensors"
        path.write_bytes(path.read_bytes()[:-1] + b"x")
        with self.assertRaisesRegex(PipelineError, "changed"):
            self.check()

    def test_checkpoint_count_must_match_even_if_manifest_claims_correct_count(self):
        self.lock["expected_parameters"] = 7
        self.manifest.update(parameters=7, trainable_parameters=7)
        with self.assertRaisesRegex(PipelineError, "tensor count"):
            self.check()

    def test_checkpoint_paths_cannot_escape_run_directory(self):
        self.manifest["checkpoint_files"] = {"../model.safetensors": "a" * 64}
        with self.assertRaisesRegex(PipelineError, "filename"):
            self.check()

    def test_checkpoint_config_must_match_locked_architecture(self):
        write_json(self.run / "checkpoint" / "config.json", dict(self.lock["architecture"], hidden_size=99))
        with self.assertRaisesRegex(PipelineError, "architecture changed"):
            self.check()


if __name__ == "__main__":
    unittest.main()
