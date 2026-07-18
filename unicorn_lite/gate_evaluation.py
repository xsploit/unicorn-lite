from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .encoder import Encoder, encoder_identity, prepare_encoder_text
from .gate_training import _is_direct, _metrics, _run_sequence
from .response_gate import (
    PonderResponseGate,
    calibrated_reply_probability,
    policy_features,
    response_gate_from_checkpoint,
)


def evaluate_response_gate(
    policy_path: str | Path,
    checkpoint_path: str | Path,
    encoder: Encoder,
    device: str = "cpu",
    output_path: str | Path | None = None,
    forced_ponder_steps: int | None = None,
    include_examples: bool = False,
    threshold_override: float | None = None,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in Path(policy_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("action") in {"OBSERVE", "REPLY_FLASH"}]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint_encoder_id = checkpoint.get("encoder_id")
    if checkpoint_encoder_id and str(checkpoint_encoder_id) != encoder_identity(encoder):
        raise ValueError("response gate checkpoint was trained with a different encoder")
    feature_dim = int(
        checkpoint.get(
            "feature_dim",
            checkpoint["model"]["structural_action_head.weight"].shape[1],
        )
    )
    model = response_gate_from_checkpoint(checkpoint).to(device)
    inputs = [
        prepare_encoder_text(
            encoder, str(row.get("model_input") or row["content"])
        )
        for row in rows
    ]
    embeddings = torch.from_numpy(encoder.encode_many(inputs, batch_size=64))
    features = torch.from_numpy(
        np.stack([policy_features(row, feature_dim) for row in rows])
    )
    labels = np.asarray(
        [row["action"] == "REPLY_FLASH" for row in rows], dtype=np.int64
    )
    ponder_diagnostics: list[dict[str, object]] = []
    probabilities, _ = _run_sequence(
        model,
        embeddings,
        features,
        rows,
        device,
        diagnostics=ponder_diagnostics,
        forced_ponder_steps=forced_ponder_steps,
    )
    calibrator = checkpoint.get("refractory_calibrator")
    if calibrator:
        probabilities = np.asarray(
            [
                calibrated_reply_probability(probability, row, calibrator)
                for probability, row in zip(probabilities, rows, strict=True)
            ],
            dtype=np.float32,
        )
    checkpoint_threshold = float(checkpoint["threshold"])
    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else checkpoint_threshold
    )

    def sliced(mask: np.ndarray) -> dict[str, Any] | None:
        if not bool(mask.any()):
            return None
        metrics = _metrics(labels[mask], probabilities[mask], threshold)
        metrics["examples"] = int(mask.sum())
        metrics["mean_probability"] = float(probabilities[mask].mean())
        return metrics

    direct = np.asarray([_is_direct(row) for row in rows])
    report: dict[str, Any] = {
        "policy": str(Path(policy_path).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "feature_dim": feature_dim,
        "threshold": threshold,
        "checkpoint_threshold": checkpoint_threshold,
        "threshold_overridden": threshold_override is not None,
        "refractory_calibrator": calibrator,
        "forced_ponder_steps": forced_ponder_steps,
        "overall": sliced(np.ones(len(rows), dtype=bool)),
        "direct": sliced(direct),
        "implicit": sliced(~direct),
    }
    if include_examples:
        report["examples"] = [
            {
                "event_id": row.get("event_id") or row.get("discord_message_id"),
                "content": row.get("content"),
                "expected": "REPLY" if bool(label) else "SILENCE",
                "probability": float(probability),
                "predicted": "REPLY" if probability >= threshold else "SILENCE",
                **(
                    {
                        "steps": int(ponder_diagnostics[index]["steps"]),
                        "probability_path": ponder_diagnostics[index]["probability_path"],
                    }
                    if ponder_diagnostics
                    else {}
                ),
            }
            for index, (row, label, probability) in enumerate(
                zip(rows, labels, probabilities, strict=True)
            )
        ]
    if isinstance(model, PonderResponseGate):
        step_values = np.asarray(
            [int(item["steps"]) for item in ponder_diagnostics], dtype=np.int64
        )
        report["ponder"] = {
            "mean_steps": float(step_values.mean()),
            "step_distribution": {
                str(step): int((step_values == step).sum())
                for step in range(1, model.max_steps + 1)
            },
        }
    categories = sorted(
        {str(row.get("external_category")) for row in rows if row.get("external_category")}
    )
    if categories:
        report["categories"] = {
            category: sliced(
                np.asarray(
                    [str(row.get("external_category")) == category for row in rows]
                )
            )
            for category in categories
        }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
