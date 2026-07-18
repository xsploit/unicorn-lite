from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    dimension: int

    def encode(self, text: str) -> np.ndarray: ...

    def encode_many(self, texts: list[str], batch_size: int = 64) -> np.ndarray: ...


class HashEncoder:
    """Offline deterministic fallback used by tests and zero-download demos."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9_']+", text.lower()) or ["<empty>"]
        for token in tokens:
            digest = hashlib.shake_256(token.encode("utf-8")).digest(self.dimension)
            values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            vector += np.where(values >= 128, 1.0, -1.0)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def encode_many(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        del batch_size
        return np.stack([self.encode(text) for text in texts])


class LocalSentenceEncoder:
    """Small local encoder. It downloads once, then runs entirely on-device."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> np.ndarray:
        result = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(result, dtype=np.float32)

    def encode_many(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        result = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return np.asarray(result, dtype=np.float32)


def _conversation_lines(text: str) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    for raw_line in text.splitlines() or [text]:
        match = re.match(r"\[([^]]+)]\s*(.*)", raw_line.strip())
        if match:
            lines.append((match.group(1), match.group(2)))
        elif raw_line.strip():
            lines.append(("CURRENT", raw_line.strip()))
    return lines or [("CURRENT", "<empty>")]


class BertConversationEncoder:
    """Frozen BERT-base control encoder using utterance-level CLS markers."""

    preserve_structure = True

    def __init__(
        self,
        model_name: str = "google-bert/bert-base-uncased",
        device: str = "cuda",
        max_length: int = 256,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.dimension = int(self._model.config.hidden_size)

    def _batch(self, texts: list[str]) -> tuple[object, object, object]:
        import torch

        rows: list[list[int]] = []
        for text in texts:
            tokens: list[int] = []
            for _, utterance in _conversation_lines(text):
                tokens.append(self.tokenizer.cls_token_id)
                tokens.extend(
                    self.tokenizer.encode(
                        utterance,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.max_length - 1,
                    )
                )
            tokens.append(self.tokenizer.sep_token_id)
            # Retain the current utterance and final separator when context is long.
            tokens = tokens[-self.max_length :]
            if tokens[0] != self.tokenizer.cls_token_id:
                tokens[0] = self.tokenizer.cls_token_id
            rows.append(tokens)
        width = max(len(row) for row in rows)
        input_ids = torch.full(
            (len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention[index, : len(row)] = 1
        return input_ids.to(self.device), attention.to(self.device), None

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        import torch

        input_ids, attention, _ = self._batch(texts)
        with torch.no_grad():
            output = self._model(input_ids=input_ids, attention_mask=attention)
            # The final utterance CLS attends bidirectionally to the conversation.
            is_cls = input_ids.eq(self.tokenizer.cls_token_id)
            positions = is_cls.long().sum(dim=1) - 1
            indices = is_cls.long().cumsum(dim=1).eq(positions[:, None] + 1).float()
            indices = indices * is_cls.float()
            vector = (output.last_hidden_state * indices.unsqueeze(-1)).sum(dim=1)
            vector = torch.nn.functional.normalize(vector, dim=-1)
        return vector.cpu().numpy().astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        return self._encode_batch([text])[0]

    def encode_many(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        outputs = [
            self._encode_batch(texts[offset : offset + batch_size])
            for offset in range(0, len(texts), batch_size)
        ]
        return np.concatenate(outputs, axis=0)


class MPCBertConversationEncoder(BertConversationEncoder):
    """PyTorch inference port of MPC-BERT with learned speaker embeddings."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda", max_length: int = 256) -> None:
        import torch
        from transformers import AutoTokenizer, BertConfig, BertModel

        self.model_name = "JasonForJoy/MPC-BERT"
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("architecture") != "mpcbert_encoder_v1":
            raise ValueError("not a converted MPC-BERT encoder checkpoint")
        config = BertConfig(**saved["config"])
        self._model = BertModel(config).to(self.device)
        self._model.load_state_dict(saved["model"])
        self._model.eval()
        self.speaker_embedding = torch.nn.Embedding(16, config.hidden_size).to(self.device)
        self.speaker_embedding.weight.data.copy_(saved["speaker_embedding"].to(self.device))
        self.speaker_embedding.eval()
        self.dimension = int(config.hidden_size)

    @staticmethod
    def _speaker(role: str) -> int:
        role = role.upper()
        if role.startswith("AGENT"):
            return 1
        if role.startswith("CURRENT"):
            return 2
        if role.startswith("OTHER_BOT"):
            return 3
        return 4

    def _batch(self, texts: list[str]) -> tuple[object, object, object]:
        import torch

        rows: list[list[int]] = []
        speakers: list[list[int]] = []
        for text in texts:
            tokens: list[int] = []
            speaker_ids: list[int] = []
            for role, utterance in _conversation_lines(text):
                utterance_ids = [self.tokenizer.cls_token_id]
                utterance_ids.extend(
                    self.tokenizer.encode(
                        utterance,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.max_length - 1,
                    )
                )
                tokens.extend(utterance_ids)
                speaker_ids.extend([self._speaker(role)] * len(utterance_ids))
            tokens.append(self.tokenizer.sep_token_id)
            speaker_ids.append(speaker_ids[-1] if speaker_ids else 2)
            tokens = tokens[-self.max_length :]
            speaker_ids = speaker_ids[-self.max_length :]
            if tokens[0] != self.tokenizer.cls_token_id:
                tokens[0] = self.tokenizer.cls_token_id
            rows.append(tokens)
            speakers.append(speaker_ids)
        width = max(len(row) for row in rows)
        input_ids = torch.full(
            (len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention = torch.zeros_like(input_ids)
        speaker_tensor = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention[index, : len(row)] = 1
            speaker_tensor[index, : len(row)] = torch.tensor(speakers[index])
        return (
            input_ids.to(self.device),
            attention.to(self.device),
            speaker_tensor.to(self.device),
        )

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        import torch

        input_ids, attention, speaker_ids = self._batch(texts)
        with torch.no_grad():
            word = self._model.embeddings.word_embeddings(input_ids)
            inputs_embeds = word + self.speaker_embedding(speaker_ids)
            output = self._model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention,
                token_type_ids=torch.zeros_like(input_ids),
            )
            is_cls = input_ids.eq(self.tokenizer.cls_token_id)
            positions = is_cls.long().sum(dim=1) - 1
            indices = is_cls.long().cumsum(dim=1).eq(positions[:, None] + 1).float()
            indices = indices * is_cls.float()
            vector = (output.last_hidden_state * indices.unsqueeze(-1)).sum(dim=1)
            vector = torch.nn.functional.normalize(vector, dim=-1)
        return vector.cpu().numpy().astype(np.float32)


def prepare_encoder_text(encoder: Encoder, text: str) -> str:
    if bool(getattr(encoder, "preserve_structure", False)):
        return text
    # Imported lazily to avoid an encoder -> response-gate module cycle.
    from .response_gate import normalize_policy_text

    return normalize_policy_text(text)


def encoder_identity(encoder: Encoder) -> str:
    """Stable checkpoint tag so equal-width encoders cannot be interchanged."""
    model_name = getattr(encoder, "model_name", None)
    if model_name:
        return str(model_name)
    return f"{type(encoder).__module__}.{type(encoder).__name__}:{encoder.dimension}"


def make_encoder(kind: str = "local", device: str = "cuda") -> Encoder:
    if kind == "hash":
        return HashEncoder()
    if kind == "local":
        return LocalSentenceEncoder(device=device)
    if kind == "bert-base":
        return BertConversationEncoder(device=device)
    if kind == "mpc-bert":
        default = Path(__file__).resolve().parents[1] / "data" / "mpcbert-encoder.pt"
        checkpoint = Path(os.getenv("UNICORN_MPCBERT_CHECKPOINT", str(default)))
        return MPCBertConversationEncoder(checkpoint, device=device)
    raise ValueError(f"unknown encoder: {kind}")
