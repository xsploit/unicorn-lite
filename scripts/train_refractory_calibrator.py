from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from unicorn_lite.encoder import make_encoder, prepare_encoder_text
from unicorn_lite.gate_training import _run_sequence
from unicorn_lite.response_gate import (
    policy_features,
    refractory_features,
    response_gate_from_checkpoint,
)


def train_calibrator(
    policy_path: str | Path,
    refractory_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    encoder_name: str = "mpc-bert",
    device: str = "cuda",
    steps: int = 600,
    learning_rate: float = 0.04,
    near_only: bool = False,
) -> dict[str, object]:
    policy_rows = [
        json.loads(line)
        for line in Path(policy_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    policy_rows = [
        row for row in policy_rows if row.get("action") in {"OBSERVE", "REPLY_FLASH"}
    ]
    train_end = int(len(policy_rows) * 0.70)
    base_rows = policy_rows[:train_end]
    refractory_rows = [
        json.loads(line)
        for line in Path(refractory_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = sorted(
        [*base_rows, *refractory_rows], key=lambda row: str(row.get("occurred_at") or "")
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if int(checkpoint["feature_dim"]) != 22:
        raise ValueError("refractory calibration expects the accepted 22-feature gate")
    encoder = make_encoder(encoder_name, device=device)
    model = response_gate_from_checkpoint(checkpoint).to(device)
    inputs = [
        prepare_encoder_text(encoder, str(row.get("model_input") or row["content"]))
        for row in rows
    ]
    embeddings = torch.from_numpy(encoder.encode_many(inputs, batch_size=64))
    features = torch.from_numpy(
        np.stack([policy_features(row, model.feature_dim) for row in rows])
    )
    base_probabilities, _ = _run_sequence(
        model, embeddings, features, rows, device
    )
    base_logits = torch.logit(
        torch.as_tensor(base_probabilities, dtype=torch.float32, device=device).clamp(
            1e-6, 1.0 - 1e-6
        )
    )
    refractory_inputs = torch.as_tensor(
        np.stack([refractory_features(row) for row in rows]),
        dtype=torch.float32,
        device=device,
    )
    if near_only:
        refractory_inputs = refractory_inputs * torch.as_tensor(
            [0.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=device
        )
    labels = torch.as_tensor(
        [row["action"] == "REPLY_FLASH" for row in rows],
        dtype=torch.float32,
        device=device,
    )
    example_weights = torch.as_tensor(
        [
            3.0
            if row.get("label_source") == "training_only_refractory_repeat"
            else 2.0
            if row.get("label_source") == "training_only_refractory_followup"
            else 1.0
            for row in rows
        ],
        dtype=torch.float32,
        device=device,
    )
    weights = torch.zeros(
        refractory_inputs.shape[1], dtype=torch.float32, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([weights], lr=learning_rate)
    initial_loss = 0.0
    final_loss = 0.0
    for step in range(steps):
        adjusted_logits = base_logits + refractory_inputs @ weights
        losses = F.binary_cross_entropy_with_logits(
            adjusted_logits, labels, reduction="none"
        )
        loss = (losses * example_weights).mean() + 0.005 * weights.square().mean()
        if step == 0:
            initial_loss = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            weights.clamp_(-6.0, 6.0)
        final_loss = float(loss.detach())

    learned = [float(value) for value in weights.detach().cpu()]
    output_checkpoint = dict(checkpoint)
    output_checkpoint["refractory_calibrator"] = {
        "version": "additive_logit_v1",
        "features": [
            "recent_answered_similarity",
            "recent_answered_near_duplicate",
            "freshness",
            "similarity_x_freshness",
        ],
        "weights": learned,
        "training_rows": len(rows),
        "refractory_rows": len(refractory_rows),
        "near_only": near_only,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, destination)
    report = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "output": str(destination.resolve()),
        "base_training_rows": len(base_rows),
        "refractory_training_rows": len(refractory_rows),
        "weights": learned,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "steps": steps,
        "learning_rate": learning_rate,
        "near_only": near_only,
    }
    destination.with_suffix(".metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tiny additive repeat cooldown on a frozen response gate"
    )
    parser.add_argument("policy_jsonl")
    parser.add_argument("refractory_jsonl")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", default="mpc-bert")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument(
        "--near-only",
        action="store_true",
        help="Train only the recent near-duplicate cooldown weight",
    )
    args = parser.parse_args()
    train_calibrator(
        args.policy_jsonl,
        args.refractory_jsonl,
        args.checkpoint,
        args.output,
        encoder_name=args.encoder,
        device=args.device,
        steps=args.steps,
        learning_rate=args.learning_rate,
        near_only=args.near_only,
    )


if __name__ == "__main__":
    main()
