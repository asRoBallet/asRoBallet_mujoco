import argparse
from collections import Counter
import logging
import os
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import evaluate, train


class DeterministicPolicy(nn.Module):
    """Export the deterministic policy mean without runtime action clipping."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observations):
        features = self.policy.extract_features(observations)
        latent = self.policy.mlp_extractor.forward_actor(features)
        return self.policy.action_net(latent)


def validate_policy_compatibility(policy):
    if policy.squash_output:
        raise ValueError(
            "ONNX export does not support squashed SB3 policies. Retrain with "
            "squash_output=False."
        )


def operator_counts(output_path):
    model = onnx.load(output_path)
    return Counter(node.op_type for node in model.graph.node)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export an asRoBallet PPO policy to ONNX."
    )
    parser.add_argument("task", choices=train.TASKS.keys(), help="Task to export.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="SB3 PPO .zip path. Defaults to the newest best checkpoint.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output .onnx path. Defaults beside the input model.",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=18,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = evaluate.default_model_path(args.task)
    if args.output_path is None:
        args.output_path = str(Path(args.model_path).with_suffix(".onnx"))
    if args.opset_version < 17:
        parser.error("--opset-version must be at least 17.")
    return args


def export_policy(
    model_path,
    output_path,
    opset_version=18,
    force=False,
):
    from stable_baselines3 import PPO

    model_path = Path(model_path)
    output_path = Path(output_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --force to overwrite it."
        )
    external_data_path = Path(f"{output_path}.data")
    if external_data_path.exists() and not force:
        raise FileExistsError(
            f"External data file already exists: {external_data_path}. "
            "Use --force to replace the previous export."
        )

    model = PPO.load(model_path, device="cpu")
    validate_policy_compatibility(model.policy)
    policy = DeterministicPolicy(model.policy).eval()
    observation_shape = model.observation_space.shape
    if observation_shape is None:
        raise ValueError("The policy must use a fixed-shape observation space.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, *observation_shape), dtype=torch.float32)
    batch = torch.export.Dim("batch", min=1)
    registration_logger = logging.getLogger(
        "torch.onnx._internal.exporter._registration"
    )
    previous_level = registration_logger.level
    registration_logger.setLevel(logging.ERROR)
    temporary_file = tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()
    try:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
                    category=FutureWarning,
                )
                torch.onnx.export(
                    policy,
                    (example,),
                    temporary_path,
                    input_names=["observations"],
                    output_names=["actions"],
                    dynamic_shapes={"observations": {0: batch}},
                    opset_version=opset_version,
                    dynamo=True,
                    external_data=False,
                )
        finally:
            registration_logger.setLevel(previous_level)

        exported = onnx.load(temporary_path)
        onnx.checker.check_model(exported)
        os.replace(temporary_path, output_path)
        if external_data_path.exists():
            external_data_path.unlink()
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return model, policy


def verify_export(model, output_path, sample_count=16, seed=3407):
    rng = np.random.default_rng(seed)
    observations = rng.normal(
        size=(sample_count, *model.observation_space.shape)
    ).astype(np.float32)
    expected_actions, _ = model.predict(observations, deterministic=True)
    policy = DeterministicPolicy(model.policy).eval()
    with torch.no_grad():
        expected_raw_actions = policy(torch.as_tensor(observations)).numpy()

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    actual_raw_actions = session.run(
        ["actions"],
        {"observations": observations},
    )[0]
    np.testing.assert_allclose(
        actual_raw_actions,
        expected_raw_actions,
        rtol=1e-5,
        atol=1e-6,
    )
    clipped_actions = np.clip(
        actual_raw_actions,
        model.action_space.low,
        model.action_space.high,
    )
    np.testing.assert_allclose(
        clipped_actions,
        expected_actions,
        rtol=1e-5,
        atol=1e-6,
    )
    return float(np.max(np.abs(clipped_actions - expected_actions)))


def main():
    args = parse_args()
    model, _ = export_policy(
        args.model_path,
        args.output_path,
        opset_version=args.opset_version,
        force=args.force,
    )
    max_error = verify_export(model, args.output_path)
    operators = operator_counts(args.output_path)
    print(f"Exported: {args.output_path}")
    print(
        f"observations={model.observation_space.shape} "
        f"actions={model.action_space.shape} "
        f"operators={dict(operators)} "
        f"max_error={max_error:.3e}"
    )


if __name__ == "__main__":
    main()
