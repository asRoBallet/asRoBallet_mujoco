import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from stable_baselines3 import PPO

from scripts import export_onnx, train


ROOT = Path(__file__).resolve().parents[1]


class TrainingTest(unittest.TestCase):
    def test_short_training_produces_exportable_model(self):
        task_name = next(iter(train.TASKS))
        with tempfile.TemporaryDirectory() as log_root:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "train.py"),
                    task_name,
                    "--n-envs",
                    "1",
                    "--total-timesteps",
                    "1",
                    "--batch-size",
                    "64",
                    "--checkpoint-freq",
                    "1024",
                    "--device",
                    "cpu",
                    "--log-root",
                    log_root,
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            model_paths = list(Path(log_root).rglob("*.zip"))
            self.assertTrue(model_paths)
            model_path = model_paths[0]
            PPO.load(model_path, device="cpu")

            onnx_path = model_path.with_suffix(".onnx")
            model, _ = export_onnx.export_policy(model_path, onnx_path)
            self.assertTrue(onnx_path.is_file())
            self.assertFalse(Path(f"{onnx_path}.data").exists())
            export_onnx.verify_export(model, onnx_path)


if __name__ == "__main__":
    unittest.main()
