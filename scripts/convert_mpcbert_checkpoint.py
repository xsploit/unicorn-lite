"""Convert exported MPC-BERT tensors into a compact PyTorch encoder checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertModel


def torch_name(tf_name: str) -> tuple[str, bool]:
    name = tf_name.removeprefix("bert/")
    if name in {
        "embeddings/word_embeddings",
        "embeddings/position_embeddings",
        "embeddings/token_type_embeddings",
    }:
        return name.replace("/", ".") + ".weight", False
    transpose = name.endswith("/kernel")
    name = name.replace("/kernel", "/weight")
    name = name.replace("/gamma", "/weight")
    name = name.replace("/beta", "/bias")
    name = name.replace("layer_", "layer/")
    return name.replace("/", "."), transpose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz")
    parser.add_argument("config")
    parser.add_argument("output")
    args = parser.parse_args()
    config_data = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = BertConfig(**config_data)
    expected = BertModel(config).state_dict()
    converted: dict[str, torch.Tensor] = {}
    speaker_embedding: torch.Tensor | None = None
    with np.load(args.npz) as archive:
        for source in archive.files:
            value = torch.from_numpy(archive[source])
            if source == "bert/embeddings/speaker_embedding":
                speaker_embedding = value
                continue
            target, transpose = torch_name(source)
            if transpose:
                value = value.T
            if target in expected:
                if value.shape != expected[target].shape:
                    raise ValueError(
                        f"shape mismatch {source} -> {target}: "
                        f"{tuple(value.shape)} != {tuple(expected[target].shape)}"
                    )
                converted[target] = value
    missing = sorted(set(expected) - set(converted))
    if missing:
        raise ValueError(f"missing BERT inference tensors: {missing}")
    if speaker_embedding is None:
        raise ValueError("MPC-BERT speaker embedding was not exported")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "mpcbert_encoder_v1",
            "config": config_data,
            "model": converted,
            "speaker_embedding": speaker_embedding,
            "source": "JasonForJoy/MPC-BERT ACL 2021 checkpoint",
        },
        output,
    )
    print(
        f"converted {len(converted)} BERT tensors and speaker table "
        f"{tuple(speaker_embedding.shape)} to {output}"
    )


if __name__ == "__main__":
    main()
