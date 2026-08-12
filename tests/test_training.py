import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from stable_baselines3 import PPO
from torch import nn

from scripts import export_onnx, train


ROOT = Path(__file__).resolve().parents[1]


class TrainingTest(unittest.TestCase):
    def test_checkpoint_configuration_and_run_layout(self):
        with mock.patch("sys.argv", ["scripts/train.py", "station_keeping"]):
            args = train.parse_args()
        self.assertEqual(args.checkpoint_freq, 100_000)
        self.assertEqual(Path(args.xml_file), Path("robots/mjcf/scene.xml"))
        self.assertEqual(train.callback_save_freq(100_001, 8), 12_501)

        with tempfile.TemporaryDirectory() as root:
            first = train.create_run_paths(root, "station_keeping", "2026-08-12_15-30-45")
            second = train.create_run_paths(root, "station_keeping", "2026-08-12_15-30-45")
            self.assertEqual(first.run_dir.name, "2026-08-12_15-30-45")
            self.assertEqual(second.run_dir.name, "2026-08-12_15-30-45_01")
            self.assertEqual(
                {path.name for path in first.run_dir.iterdir()},
                {"tensorboard", "monitor", "checkpoints"},
            )

    def test_short_training_writes_loadable_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "train.py"),
                    "station_keeping",
                    "--n-envs",
                    "1",
                    "--total-timesteps",
                    "16",
                    "--batch-size",
                    "64",
                    "--checkpoint-freq",
                    "1024",
                    "--device",
                    "cpu",
                    "--log-root",
                    root,
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            run_dir = next((Path(root) / "station_keeping").iterdir())
            checkpoints = run_dir / "checkpoints"
            self.assertTrue((checkpoints / "model_1024_steps.zip").is_file())
            self.assertTrue((checkpoints / "best_model.zip").is_file())
            final_path = checkpoints / "final_model.zip"
            trained_model = PPO.load(final_path, device="cpu")
            self.assertTrue(
                any(
                    isinstance(module, nn.ELU)
                    for module in trained_model.policy.mlp_extractor.policy_net.modules()
                )
            )
            onnx_path = checkpoints / "final_model.onnx"
            model, _ = export_onnx.export_policy(final_path, onnx_path)
            self.assertTrue(onnx_path.is_file())
            self.assertFalse(Path(f"{onnx_path}.data").exists())
            self.assertEqual(
                export_onnx.operator_counts(onnx_path),
                {"Gemm": 3, "Elu": 2},
            )
            self.assertLess(export_onnx.verify_export(model, onnx_path), 1e-5)
            self.assertTrue(list((run_dir / "tensorboard").glob("events.out.tfevents.*")))
            self.assertTrue(list((run_dir / "monitor").glob("*.monitor.csv")))
            self.assertFalse(list(run_dir.rglob("PPO_*")))


if __name__ == "__main__":
    unittest.main()
