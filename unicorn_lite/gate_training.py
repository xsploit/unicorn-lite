from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .encoder import Encoder, encoder_identity, prepare_encoder_text
from .response_gate import (
    PonderResponseGate,
    ResponseGate,
    looks_like_question,
    policy_features,
)


REQUEST_OPENERS = {
    "calculate", "check", "choose", "compare", "convert", "create", "define",
    "explain", "find", "give", "help", "list", "look", "make", "output",
    "research", "run", "send", "show", "summarize", "tell", "use", "write",
}


def _is_direct(row: dict[str, object]) -> bool:
    return bool(
        row.get("direct")
        or row.get("mentioned_target")
        or row.get("mentioned")
        or row.get("replied_to_target")
    )


def _content_words(content: str) -> tuple[str, list[str]]:
    without_address = re.sub(r"<@!?\d+>", " ", content)
    without_address = re.sub(r"https?://\S+", " ", without_address)
    return without_address, re.findall(r"[A-Za-z0-9']+", without_address)


def _load_hard_negative_rows(
    paths: list[str | Path],
    cutoff: str,
    max_words: int = 0,
    exclude_questions: bool = False,
    exclude_requests: bool = False,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Load only pre-cutoff unanswered direct turns for training augmentation."""
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    counts = {
        "scanned": 0,
        "selected": 0,
        "after_cutoff": 0,
        "not_direct": 0,
        "too_long": 0,
        "question": 0,
        "request": 0,
    }
    for source_path in paths:
        source = Path(source_path)
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            counts["scanned"] += 1
            row: dict[str, Any] = json.loads(line)
            if row.get("action") != "OBSERVE" or not _is_direct(row):
                counts["not_direct"] += 1
                continue
            occurred_at = str(row.get("occurred_at") or "")
            if not occurred_at or occurred_at > cutoff:
                counts["after_cutoff"] += 1
                continue
            content = str(row.get("content") or "")
            content_without_address, words = _content_words(content)
            if max_words > 0 and len(words) > max_words:
                counts["too_long"] += 1
                continue
            if exclude_questions and (
                "?" in content_without_address
                or looks_like_question(content_without_address)
            ):
                counts["question"] += 1
                continue
            if exclude_requests and words and (
                words[0].casefold() in REQUEST_OPENERS
                or "please" in {word.casefold() for word in words}
            ):
                counts["request"] += 1
                continue
            identity = str(
                row.get("discord_message_id")
                or row.get("event_id")
                or f"{occurred_at}:{row.get('content')}"
            )
            if identity in seen:
                continue
            seen.add(identity)
            candidate = dict(row)
            candidate["channel"] = (
                f"hard-negative:{source.stem}:"
                f"{row.get('channel', 'default')}"
            )
            candidate["label_source"] = "training_only_unanswered_direct"
            candidate["hard_negative_source"] = source.name
            selected.append(candidate)
    selected.sort(key=lambda row: str(row.get("occurred_at") or ""))
    counts["selected"] = len(selected)
    return selected, counts


def _load_positive_anchor_rows(
    paths: list[str | Path], cutoff: str, limit: int
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Select concise pre-cutoff direct questions/requests as balance anchors."""
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    counts = {
        "scanned": 0,
        "candidates": 0,
        "selected": 0,
        "after_cutoff": 0,
        "not_direct_positive": 0,
        "not_concise_question_or_request": 0,
    }
    for source_path in paths:
        source = Path(source_path)
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            counts["scanned"] += 1
            row: dict[str, Any] = json.loads(line)
            if row.get("action") != "REPLY_FLASH" or not _is_direct(row):
                counts["not_direct_positive"] += 1
                continue
            occurred_at = str(row.get("occurred_at") or "")
            if not occurred_at or occurred_at > cutoff:
                counts["after_cutoff"] += 1
                continue
            content = str(row.get("content") or "")
            content_without_address, words = _content_words(content)
            request_like = bool(
                words
                and (
                    words[0].casefold() in REQUEST_OPENERS
                    or "please" in {word.casefold() for word in words}
                )
            )
            if not 3 <= len(words) <= 12 or not (
                "?" in content_without_address
                or looks_like_question(content_without_address)
                or request_like
            ):
                counts["not_concise_question_or_request"] += 1
                continue
            identity = str(
                row.get("discord_message_id")
                or row.get("event_id")
                or f"{occurred_at}:{content}"
            )
            if identity in seen:
                continue
            seen.add(identity)
            candidate = dict(row)
            candidate["channel"] = (
                f"positive-anchor:{source.stem}:"
                f"{row.get('channel', 'default')}"
            )
            candidate["label_source"] = "training_only_positive_anchor"
            candidate["positive_anchor_source"] = source.name
            candidates.append(candidate)
    candidates.sort(key=lambda row: str(row.get("occurred_at") or ""))
    counts["candidates"] = len(candidates)
    if 0 < limit < len(candidates):
        indices = np.linspace(0, len(candidates) - 1, limit, dtype=np.int64)
        selected = [candidates[int(index)] for index in indices]
    else:
        selected = candidates if limit != 0 else []
    counts["selected"] = len(selected)
    return selected, counts


def _load_refractory_rows(
    paths: list[str | Path], cutoff: str
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Load paired, training-only repeat negatives and follow-up controls."""
    selected: list[dict[str, object]] = []
    counts = {
        "scanned": 0,
        "selected": 0,
        "after_cutoff": 0,
        "invalid": 0,
        "pairs": 0,
    }
    pair_ids: set[str] = set()
    for source_path in paths:
        source = Path(source_path)
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            counts["scanned"] += 1
            row: dict[str, Any] = json.loads(line)
            pair_id = str(row.get("refractory_pair_id") or "")
            if (
                row.get("action") not in {"OBSERVE", "REPLY_FLASH"}
                or not pair_id
                or not str(row.get("label_source") or "").startswith(
                    "training_only_refractory_"
                )
            ):
                counts["invalid"] += 1
                continue
            occurred_at = str(row.get("occurred_at") or "")
            if not occurred_at or occurred_at > cutoff:
                counts["after_cutoff"] += 1
                continue
            candidate = dict(row)
            candidate["channel"] = f"refractory:{source.stem}:{pair_id}"
            candidate["refractory_source"] = source.name
            selected.append(candidate)
            pair_ids.add(pair_id)
    selected.sort(key=lambda row: str(row.get("occurred_at") or ""))
    counts["selected"] = len(selected)
    counts["pairs"] = len(pair_ids)
    return selected, counts


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probabilities >= threshold
    actual = labels.astype(bool)
    tp = int(np.logical_and(predicted, actual).sum())
    fp = int(np.logical_and(predicted, ~actual).sum())
    fn = int(np.logical_and(~predicted, actual).sum())
    tn = int(np.logical_and(~predicted, ~actual).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    order = np.argsort(-probabilities)
    ordered_labels = labels[order]
    cumulative = np.cumsum(ordered_labels)
    precision_at_rank = cumulative / np.arange(1, len(labels) + 1)
    average_precision = float(
        (precision_at_rank * ordered_labels).sum() / max(1, int(labels.sum()))
    )
    return {
        "accuracy": (tp + tn) / max(1, len(labels)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "average_precision": average_precision,
        "predicted_reply_rate": float(predicted.mean()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def _best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.02, 0.98, 97)
    return float(max(candidates, key=lambda value: _metrics(labels, probabilities, value)["f1"]))


def _run_sequence(
    model: ResponseGate,
    embeddings: torch.Tensor,
    features: torch.Tensor,
    rows: list[dict[str, object]],
    device: str,
    states: dict[str, torch.Tensor] | None = None,
    diagnostics: list[dict[str, object]] | None = None,
    forced_ponder_steps: int | None = None,
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    channel_states = states or {}
    probabilities: list[float] = []
    model.eval()
    with torch.no_grad():
        for index, row in enumerate(rows):
            channel = str(row.get("channel", "default"))
            hidden = channel_states.get(
                channel, torch.zeros(model.hidden_dim, device=device)
            )
            if isinstance(model, PonderResponseGate):
                output = model.ponder(
                    embeddings[index].to(device),
                    features[index].to(device),
                    hidden,
                    adaptive=forced_ponder_steps is None,
                    forced_steps=forced_ponder_steps,
                )
                hidden, logits = output.hidden, output.logits
                if diagnostics is not None:
                    diagnostics.append(
                        {
                            "steps": output.steps,
                            "halt_probability": float(output.halt_probabilities[-1]),
                            "probability_path": output.probability_path,
                        }
                    )
            else:
                hidden, logits = model.step(
                    embeddings[index].to(device), features[index].to(device), hidden
                )
            channel_states[channel] = hidden
            probabilities.append(float(F.softmax(logits, dim=-1)[1]))
    return np.asarray(probabilities, dtype=np.float32), channel_states


def _baseline_metrics(rows: list[dict[str, object]], labels: np.ndarray) -> dict[str, object]:
    zero = np.zeros(len(rows), dtype=np.float32)
    direct = np.asarray([float(_is_direct(row)) for row in rows], dtype=np.float32)
    question = np.asarray([float(bool(row.get("question"))) for row in rows], dtype=np.float32)
    direct_or_question = np.maximum(direct, question)
    return {
        "always_observe": _metrics(labels, zero, 0.5),
        "reply_if_direct": _metrics(labels, direct, 0.5),
        "reply_if_question": _metrics(labels, question, 0.5),
        "reply_if_direct_or_question": _metrics(labels, direct_or_question, 0.5),
    }


def _expand_feature_checkpoint(
    model: ResponseGate,
    source_state: dict[str, torch.Tensor],
    source_feature_dim: int,
) -> dict[str, torch.Tensor]:
    """Preserve an accepted gate while appending zero-initialized features."""
    if source_feature_dim > model.feature_dim:
        raise ValueError("initial checkpoint has more policy features than the model")
    target = model.state_dict()
    for key, source in source_state.items():
        if key not in target:
            continue
        if target[key].shape == source.shape:
            target[key] = source
            continue
        expanded = torch.zeros_like(target[key])
        if key in {"feature_project.0.weight", "structural_action_head.weight"}:
            expanded[:, :source_feature_dim] = source
        elif key == "halt_head.weight" and isinstance(model, PonderResponseGate):
            hidden_dim = model.hidden_dim
            expanded[:, :hidden_dim] = source[:, :hidden_dim]
            expanded[:, hidden_dim : hidden_dim + source_feature_dim] = source[
                :, hidden_dim:
            ]
        else:
            raise ValueError(f"cannot expand checkpoint tensor: {key}")
        target[key] = expanded
    return target


def train_response_gate(
    policy_path: str | Path,
    encoder: Encoder,
    output_path: str | Path,
    device: str = "cpu",
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 8e-4,
    seed: int = 7,
    initial_checkpoint_path: str | Path | None = None,
    architecture: str = "single",
    max_ponder_steps: int = 8,
    min_ponder_steps: int = 1,
    halt_threshold: float = 0.5,
    compute_penalty: float = 0.015,
    stability_penalty: float = 0.01,
    hard_negative_paths: list[str | Path] | None = None,
    hard_negative_max_words: int = 0,
    hard_negative_exclude_questions: bool = False,
    hard_negative_exclude_requests: bool = False,
    positive_anchor_paths: list[str | Path] | None = None,
    positive_anchor_count: int = 0,
    refractory_paths: list[str | Path] | None = None,
) -> dict[str, object]:
    all_rows = [
        json.loads(line)
        for line in Path(policy_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in all_rows if row.get("action") in {"OBSERVE", "REPLY_FLASH"}]
    if len(rows) < 100:
        raise ValueError("response-gate training needs at least 100 unambiguous rows")
    train_end = int(len(rows) * 0.70)
    calibration_end = int(len(rows) * 0.85)
    base_train_rows = rows[:train_end]
    calibration_rows = rows[train_end:calibration_end]
    test_rows = rows[calibration_end:]
    train_cutoff = max(str(row.get("occurred_at") or "") for row in base_train_rows)
    hard_negative_rows, hard_negative_report = _load_hard_negative_rows(
        list(hard_negative_paths or []),
        train_cutoff,
        max_words=hard_negative_max_words,
        exclude_questions=hard_negative_exclude_questions,
        exclude_requests=hard_negative_exclude_requests,
    )
    positive_anchor_rows, positive_anchor_report = _load_positive_anchor_rows(
        list(positive_anchor_paths or []), train_cutoff, positive_anchor_count
    )
    refractory_rows, refractory_report = _load_refractory_rows(
        list(refractory_paths or []), train_cutoff
    )
    train_rows = sorted(
        [
            *base_train_rows,
            *hard_negative_rows,
            *positive_anchor_rows,
            *refractory_rows,
        ],
        key=lambda row: str(row.get("occurred_at") or ""),
    )
    working_rows = [*train_rows, *calibration_rows, *test_rows]

    model_inputs = [
        prepare_encoder_text(
            encoder, str(row.get("model_input") or row["content"])
        )
        for row in working_rows
    ]
    embeddings = torch.from_numpy(encoder.encode_many(model_inputs, batch_size=64))
    features = torch.from_numpy(np.stack([policy_features(row) for row in working_rows]))
    labels = torch.tensor(
        [1 if row["action"] == "REPLY_FLASH" else 0 for row in working_rows],
        dtype=torch.long,
    )
    augmented_train_end = len(train_rows)
    augmented_calibration_end = augmented_train_end + len(calibration_rows)
    train_embeddings = embeddings[:augmented_train_end]
    calibration_embeddings = embeddings[augmented_train_end:augmented_calibration_end]
    test_embeddings = embeddings[augmented_calibration_end:]
    train_features = features[:augmented_train_end]
    calibration_features = features[augmented_train_end:augmented_calibration_end]
    test_features = features[augmented_calibration_end:]
    train_labels = labels[:augmented_train_end]
    calibration_labels = labels[augmented_train_end:augmented_calibration_end]
    test_labels = labels[augmented_calibration_end:]

    torch.manual_seed(seed)
    if architecture == "ponder":
        model: ResponseGate = PonderResponseGate(
            encoder.dimension,
            max_steps=max_ponder_steps,
            min_steps=min_ponder_steps,
            halt_threshold=halt_threshold,
        ).to(device)
    elif architecture == "single":
        model = ResponseGate(encoder.dimension).to(device)
    else:
        raise ValueError(f"unknown response gate architecture: {architecture}")
    initialized_from: str | None = None
    if initial_checkpoint_path is not None:
        initial_path = Path(initial_checkpoint_path)
        initial = torch.load(initial_path, map_location=device, weights_only=True)
        initial_feature_dim = int(
            initial.get(
                "feature_dim",
                initial["model"]["structural_action_head.weight"].shape[1],
            )
        )
        if int(initial["embedding_dim"]) != encoder.dimension:
            raise ValueError("initial checkpoint embedding dimension does not match encoder")
        initial_encoder_id = initial.get("encoder_id")
        if initial_encoder_id and str(initial_encoder_id) != encoder_identity(encoder):
            raise ValueError("initial checkpoint was trained with a different encoder")
        if initial_feature_dim > model.feature_dim:
            raise ValueError("initial checkpoint policy feature dimension is incompatible")
        if initial_feature_dim < model.feature_dim:
            model.load_state_dict(
                _expand_feature_checkpoint(
                    model, initial["model"], initial_feature_dim
                )
            )
        elif isinstance(model, PonderResponseGate) and initial.get("architecture") != model.architecture:
            compatible = {
                key: value
                for key, value in initial["model"].items()
                if key in model.state_dict()
                and model.state_dict()[key].shape == value.shape
            }
            model.load_state_dict(compatible, strict=False)
        else:
            model.load_state_dict(initial["model"])
        initialized_from = str(initial_path.resolve())
    positives = max(1, int(train_labels.sum()))
    negatives = max(1, len(train_labels) - positives)
    recency_weights = torch.linspace(0.35, 1.0, len(train_rows)) ** 2
    weighted_positives = float(recency_weights[train_labels == 1].sum())
    weighted_negatives = float(recency_weights[train_labels == 0].sum())
    class_weights = torch.tensor(
        [1.0, weighted_negatives / max(1e-6, weighted_positives)], device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
    best_ap = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    final_loss = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        channel_states: dict[str, torch.Tensor] = {}
        for offset in range(0, len(train_rows), batch_size):
            losses: list[torch.Tensor] = []
            end = min(len(train_rows), offset + batch_size)
            for index in range(offset, end):
                channel = str(train_rows[index].get("channel", "default"))
                hidden = channel_states.get(
                    channel, torch.zeros(model.hidden_dim, device=device)
                )
                if isinstance(model, PonderResponseGate):
                    output = model.ponder(
                        train_embeddings[index].to(device),
                        train_features[index].to(device),
                        hidden,
                        adaptive=False,
                    )
                    channel_states[channel] = output.hidden
                    per_step_losses = torch.stack(
                        [
                            F.cross_entropy(
                                logits.unsqueeze(0),
                                train_labels[index].view(1).to(device),
                                weight=class_weights,
                                label_smoothing=0.03,
                            )
                            for logits in output.logits_by_step
                        ]
                    )
                    depths = torch.arange(
                        1,
                        output.steps + 1,
                        dtype=torch.float32,
                        device=device,
                    )
                    expected_depth = (output.halt_weights * depths).sum()
                    reply_path = torch.stack(
                        [F.softmax(logits, dim=-1)[1] for logits in output.logits_by_step]
                    )
                    stability = (
                        (reply_path[1:] - reply_path[:-1]).pow(2).mean()
                        if output.steps > 1
                        else torch.zeros((), device=device)
                    )
                    event_loss = (
                        (output.halt_weights * per_step_losses).sum()
                        + compute_penalty * expected_depth / model.max_steps
                        + stability_penalty * stability
                    )
                else:
                    hidden, logits = model.step(
                        train_embeddings[index].to(device),
                        train_features[index].to(device),
                        hidden,
                    )
                    channel_states[channel] = hidden
                    event_loss = F.cross_entropy(
                        logits.unsqueeze(0),
                        train_labels[index].view(1).to(device),
                        weight=class_weights,
                        label_smoothing=0.03,
                    )
                losses.append(recency_weights[index].to(device) * event_loss)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            channel_states = {
                channel: hidden.detach() for channel, hidden in channel_states.items()
            }
            final_loss = float(loss.detach())

        _, primed_states = _run_sequence(
            model, train_embeddings, train_features, train_rows, device
        )
        calibration_probabilities, _ = _run_sequence(
            model,
            calibration_embeddings,
            calibration_features,
            calibration_rows,
            device,
            states=primed_states,
        )
        calibration_metrics = _metrics(
            calibration_labels.numpy(), calibration_probabilities, 0.5
        )
        if calibration_metrics["average_precision"] > best_ap + 1e-5:
            best_ap = calibration_metrics["average_precision"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 12:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    train_probabilities, train_states = _run_sequence(
        model, train_embeddings, train_features, train_rows, device
    )
    calibration_probabilities, calibration_states = _run_sequence(
        model,
        calibration_embeddings,
        calibration_features,
        calibration_rows,
        device,
        states=train_states,
    )
    ponder_diagnostics: list[dict[str, object]] = []
    test_probabilities, _ = _run_sequence(
        model,
        test_embeddings,
        test_features,
        test_rows,
        device,
        states=calibration_states,
        diagnostics=ponder_diagnostics,
    )
    threshold = _best_threshold(
        calibration_labels.numpy(), calibration_probabilities
    )
    calibration_metrics = _metrics(
        calibration_labels.numpy(), calibration_probabilities, threshold
    )
    test_metrics = _metrics(
        test_labels.numpy(), test_probabilities, threshold
    )
    baselines = _baseline_metrics(test_rows, test_labels.numpy())
    direct_baseline = baselines["reply_if_direct"]
    precision_efficient = bool(
        test_metrics["average_precision"] > direct_baseline["average_precision"]
        and test_metrics["precision"] > direct_baseline["precision"]
        and test_metrics["recall"] >= 0.35
        and test_metrics["predicted_reply_rate"] < direct_baseline["predicted_reply_rate"]
    )
    initiative_balanced = bool(
        test_metrics["average_precision"]
        >= direct_baseline["average_precision"] * 0.98
        and test_metrics["f1"] >= direct_baseline["f1"] * 0.98
        and test_metrics["recall"] > direct_baseline["recall"]
        and test_metrics["predicted_reply_rate"] <= 0.30
    )
    acceptance = precision_efficient or initiative_balanced
    acceptance_profile = (
        "precision_efficient"
        if precision_efficient
        else "initiative_balanced"
        if initiative_balanced
        else None
    )

    report: dict[str, object] = {
        "examples_total": len(all_rows),
        "examples_used": len(rows),
        "unknown_excluded": len(all_rows) - len(rows),
        "train_examples": len(train_rows),
        "base_train_examples": len(base_train_rows),
        "hard_negative_examples": len(hard_negative_rows),
        "hard_negative_selection": hard_negative_report,
        "hard_negative_manifest": [
            {
                "discord_message_id": row.get("discord_message_id"),
                "occurred_at": row.get("occurred_at"),
                "content": row.get("content"),
                "source": row.get("hard_negative_source"),
            }
            for row in hard_negative_rows
        ],
        "positive_anchor_examples": len(positive_anchor_rows),
        "positive_anchor_selection": positive_anchor_report,
        "positive_anchor_manifest": [
            {
                "discord_message_id": row.get("discord_message_id"),
                "occurred_at": row.get("occurred_at"),
                "content": row.get("content"),
                "source": row.get("positive_anchor_source"),
            }
            for row in positive_anchor_rows
        ],
        "refractory_examples": len(refractory_rows),
        "refractory_selection": refractory_report,
        "refractory_manifest": [
            {
                "refractory_pair_id": row.get("refractory_pair_id"),
                "occurred_at": row.get("occurred_at"),
                "content": row.get("content"),
                "action": row.get("action"),
                "label_source": row.get("label_source"),
                "source": row.get("refractory_source"),
            }
            for row in refractory_rows
        ],
        "train_cutoff": train_cutoff,
        "calibration_examples": len(calibration_rows),
        "test_examples": len(test_rows),
        "train_replies": int(train_labels.sum()),
        "calibration_replies": int(calibration_labels.sum()),
        "test_replies": int(test_labels.sum()),
        "best_epoch": best_epoch,
        "epochs_ran": best_epoch + stale_epochs,
        "policy_mode": "learned gate only; pings are features, never obligations",
        "architecture": (
            model.architecture if isinstance(model, PonderResponseGate) else "single_v1"
        ),
        "encoder_id": encoder_identity(encoder),
        "initialized_from": initialized_from,
        "initial_feature_dim": (
            initial_feature_dim if initial_checkpoint_path is not None else None
        ),
        "threshold_selected_on_calibration": threshold,
        "recency_weighting": {
            "oldest": float(recency_weights[0]),
            "newest": float(recency_weights[-1]),
            "effective_positive_class_weight": float(class_weights[1]),
        },
        "final_batch_loss": final_loss,
        "calibration": calibration_metrics,
        "test": test_metrics,
        "baselines": baselines,
        "acceptance_passed": acceptance,
        "acceptance_profile": acceptance_profile,
        "acceptance_checks": {
            "precision_efficient": precision_efficient,
            "initiative_balanced": initiative_balanced,
        },
        "deployment": "eligible for shadow evaluation" if acceptance else "offline only",
    }
    if isinstance(model, PonderResponseGate):
        step_values = np.asarray(
            [int(item["steps"]) for item in ponder_diagnostics], dtype=np.int64
        )
        report["ponder"] = {
            "max_steps": model.max_steps,
            "min_steps": model.min_steps,
            "halt_threshold": model.halt_threshold,
            "compute_penalty": compute_penalty,
            "stability_penalty": stability_penalty,
            "test_mean_steps": float(step_values.mean()),
            "test_step_distribution": {
                str(step): int((step_values == step).sum())
                for step in range(1, model.max_steps + 1)
            },
        }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": (
                model.architecture if isinstance(model, PonderResponseGate) else "single_v1"
            ),
            "embedding_dim": encoder.dimension,
            "encoder_id": encoder_identity(encoder),
            "hard_negative_count": len(hard_negative_rows),
            "positive_anchor_count": len(positive_anchor_rows),
            "refractory_count": len(refractory_rows),
            "hidden_dim": model.hidden_dim,
            "feature_dim": model.feature_dim,
            "threshold": threshold,
            "policy_mode": "learned_gate_only",
            "acceptance_passed": acceptance,
            "acceptance_profile": acceptance_profile,
            "ponder": (
                {
                    "max_steps": model.max_steps,
                    "min_steps": model.min_steps,
                    "halt_threshold": model.halt_threshold,
                    "compute_penalty": compute_penalty,
                    "stability_penalty": stability_penalty,
                }
                if isinstance(model, PonderResponseGate)
                else None
            ),
        },
        output,
    )
    report["checkpoint"] = str(output.resolve())
    output.with_suffix(".metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
