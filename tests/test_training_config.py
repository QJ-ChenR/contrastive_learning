"""CLI/configuration boundaries that protect an experiment's protocol."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from dna_training.cli import main
from dna_training.config import PROJECT_ROOT, TrainingConfig, load_config
from dna_training.runtime import evaluate_run
from test_training_data import cohort_fixture


class TrainingConfigurationTests(unittest.TestCase):
    def test_typos_and_invalid_resource_settings_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({"hed_lr": 0.001}))
            with self.assertRaisesRegex(ValueError, "Unknown configuration"):
                load_config(path)
        for overrides in ({"max_tokens": 2}, {"max_tokens": True}, {"head_lr": float("nan")},
                          {"gradient_replay": "false"}, {"lora_targets": "attn.Wqkv"},
                          {"device": "cpu"}, {"warmup_ratio": 1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                load_config(overrides=overrides)

    def test_resume_inherits_saved_protocol_before_cpu_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            config = TrainingConfig(device="cpu", dtype="float32", attention_backend="eager", mode="lora",
                                    seed=77, max_epochs=11, output_dir=str(output))
            (output / "config.json").write_text(json.dumps(config.to_dict()))
            with mock.patch("dna_training.runtime.train", return_value={}) as train, redirect_stdout(io.StringIO()):
                main(["train", "--resume", "--output-dir", str(output), "--device", "cpu"])
            restored, resume, _ = train.call_args.args
            self.assertTrue(resume)
            self.assertEqual((restored.mode, restored.seed, restored.max_epochs, restored.dtype),
                             ("lora", 77, 11, "float32"))

    def test_evaluation_cannot_change_selection_protocol(self):
        with self.assertRaisesRegex(ValueError, "saved training or metric protocol"):
            evaluate_run("/nonexistent/run", overrides={"eval_k": 1})
        with self.assertRaisesRegex(ValueError, "saved training or metric protocol"):
            evaluate_run("/nonexistent/run", overrides={"mode": "frozen"})

    def test_script_runs_from_another_directory_and_audit_needs_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split, matrix = cohort_fixture(root)
            config = root / "config.json"
            config.write_text(json.dumps({"project_root": str(root), "split_dir": str(split),
                                          "matrix_path": str(matrix)}))
            result = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/train_dna_encoder.py"),
                                     "audit", "--config", str(config)], cwd=root,
                                    text=True, capture_output=True, check=True)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["epoch_0"]["unique_anchors"], 4)
            self.assertFalse(report["heldout_relations_used_for_sampling"])


if __name__ == "__main__":
    unittest.main()
