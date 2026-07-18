from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .models import MemoryHit


ARCHITECTURE = "discord_emdr2_v1"


def emdr2_retriever_loss(
    retriever_scores: torch.Tensor,
    answer_log_likelihood: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Equation 6's retriever term over an already selected top-K set.

    The answer likelihood is deliberately detached. The shared reader teaches the
    retriever which memories support the answer, but this term cannot train the
    reader to become an easier teacher.
    """
    if retriever_scores.shape != answer_log_likelihood.shape:
        raise ValueError("retriever scores and answer likelihoods must align")
    log_prior = F.log_softmax(retriever_scores / temperature, dim=-1)
    joint = log_prior + answer_log_likelihood.detach()
    return -torch.logsumexp(joint, dim=-1).mean()


def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    return F.normalize(pooled, dim=-1)


def reader_document(query: str, actor: str, memory: str) -> str:
    return f"question: {query}\ncontext from {actor}: {memory}"


@dataclass(slots=True)
class EMDR2Step:
    loss: torch.Tensor
    reader_loss: torch.Tensor
    retriever_loss: torch.Tensor
    answer_log_likelihood: torch.Tensor
    retriever_scores: torch.Tensor
    selected_indices: torch.Tensor


class DiscordEMDR2(nn.Module):
    """Small shared-reader EMDR2 model for causal Discord memory groups."""

    def __init__(
        self,
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reader_model: str = "google/flan-t5-small",
        *,
        device: str = "cuda",
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

        self.device_name = device
        self.retriever_model_name = retriever_model
        self.reader_model_name = reader_model
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model)
        self.reader_tokenizer = AutoTokenizer.from_pretrained(reader_model)
        self.query_encoder = AutoModel.from_pretrained(retriever_model)
        self.memory_encoder = AutoModel.from_pretrained(retriever_model)
        self.reader = AutoModelForSeq2SeqLM.from_pretrained(reader_model)
        if gradient_checkpointing:
            for model in (self.query_encoder, self.memory_encoder, self.reader):
                enable = getattr(model, "gradient_checkpointing_enable", None)
                if enable is not None:
                    enable()
            if hasattr(self.reader.config, "use_cache"):
                self.reader.config.use_cache = False
        self.to(device)

    @property
    def retriever_dimension(self) -> int:
        return int(self.query_encoder.config.hidden_size)

    def _retriever_tokens(
        self, texts: list[str], max_length: int
    ) -> dict[str, torch.Tensor]:
        values = self.retriever_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device_name) for key, value in values.items()}

    def encode_queries(self, texts: list[str], max_length: int = 128) -> torch.Tensor:
        tokens = self._retriever_tokens(texts, max_length)
        output = self.query_encoder(**tokens)
        return _mean_pool(output.last_hidden_state, tokens["attention_mask"])

    def encode_memories(self, texts: list[str], max_length: int = 192) -> torch.Tensor:
        tokens = self._retriever_tokens(texts, max_length)
        output = self.memory_encoder(**tokens)
        return _mean_pool(output.last_hidden_state, tokens["attention_mask"])

    def score_candidates(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        max_length: int = 192,
    ) -> torch.Tensor:
        query_vector = self.encode_queries([query], max_length)
        memory_vectors = self.encode_memories(
            [str(candidate["content"]) for candidate in candidates], max_length
        )
        return (query_vector @ memory_vectors.T)[0]

    def _reader_inputs(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        indices: Iterable[int],
        max_source_length: int,
    ) -> dict[str, torch.Tensor]:
        texts = [
            reader_document(
                query,
                str(candidates[index].get("actor") or "unknown"),
                str(candidates[index]["content"]),
            )
            for index in indices
        ]
        values = self.reader_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device_name) for key, value in values.items()}

    def _labels(self, reply: str, max_target_length: int) -> torch.Tensor:
        values = self.reader_tokenizer(
            text_target=[reply],
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
        )["input_ids"].to(self.device_name)
        return values.masked_fill(values == self.reader_tokenizer.pad_token_id, -100)

    @torch.no_grad()
    def answer_log_likelihood(
        self,
        *,
        query: str,
        reply: str,
        candidates: list[dict[str, Any]],
        indices: Iterable[int],
        max_source_length: int = 160,
        max_target_length: int = 96,
    ) -> torch.Tensor:
        from transformers.modeling_outputs import BaseModelOutput

        selected = list(indices)
        reader_inputs = self._reader_inputs(
            query, candidates, selected, max_source_length
        )
        labels = self._labels(reply, max_target_length).repeat(len(selected), 1)
        hidden = self.reader.get_encoder()(**reader_inputs).last_hidden_state
        output = self.reader(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=reader_inputs["attention_mask"],
            labels=labels,
        )
        logits = F.log_softmax(output.logits.float(), dim=-1)
        safe_labels = labels.masked_fill(labels < 0, 0)
        token_values = torch.gather(
            logits, dim=-1, index=safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        return (token_values * labels.ne(-100)).sum(dim=-1)

    def training_step(
        self,
        *,
        query: str,
        reply: str,
        candidates: list[dict[str, Any]],
        top_k: int = 4,
        retriever_max_length: int = 192,
        reader_max_source_length: int = 160,
        reader_max_target_length: int = 96,
        retriever_temperature: float = 1.0,
    ) -> EMDR2Step:
        if not candidates:
            raise ValueError("EMDR2 needs at least one candidate memory")
        all_scores = self.score_candidates(query, candidates, retriever_max_length)
        selected_count = min(int(top_k), len(candidates))
        selected_scores, selected_indices = torch.topk(
            all_scores, k=selected_count
        )

        selected_list = [int(index) for index in selected_indices.detach().cpu()]
        reader_inputs = self._reader_inputs(
            query, candidates, selected_list, reader_max_source_length
        )
        labels = self._labels(reply, reader_max_target_length)

        encoder_output = self.reader.get_encoder()(**reader_inputs)
        hidden = encoder_output.last_hidden_state
        source_mask = reader_inputs["attention_mask"]
        # FiD: each memory is encoded independently, then all token states are
        # concatenated before the one shared decoder predicts the reply.
        fused_hidden = hidden.reshape(1, -1, hidden.shape[-1])
        fused_mask = source_mask.reshape(1, -1)
        from transformers.modeling_outputs import BaseModelOutput

        reader_output = self.reader(
            encoder_outputs=BaseModelOutput(last_hidden_state=fused_hidden),
            attention_mask=fused_mask,
            labels=labels,
        )
        reader_loss = reader_output.loss

        # E-step teacher: score the gold reply once per individual memory with
        # the exact same reader parameters. No activations are retained because
        # Equation 6 applies stop-gradient to these likelihoods.
        with torch.no_grad():
            repeated_labels = labels.repeat(selected_count, 1)
            individual_output = self.reader(
                encoder_outputs=BaseModelOutput(last_hidden_state=hidden.detach()),
                attention_mask=source_mask,
                labels=repeated_labels,
            )
            logits = F.log_softmax(individual_output.logits.float(), dim=-1)
            safe_labels = repeated_labels.masked_fill(repeated_labels < 0, 0)
            token_log_likelihood = torch.gather(
                logits, dim=-1, index=safe_labels.unsqueeze(-1)
            ).squeeze(-1)
            label_mask = repeated_labels.ne(-100)
            answer_log_likelihood = (
                token_log_likelihood * label_mask
            ).sum(dim=-1)

        retriever_loss = emdr2_retriever_loss(
            selected_scores.unsqueeze(0),
            answer_log_likelihood.unsqueeze(0),
            temperature=retriever_temperature,
        )
        return EMDR2Step(
            loss=reader_loss + retriever_loss,
            reader_loss=reader_loss,
            retriever_loss=retriever_loss,
            answer_log_likelihood=answer_log_likelihood,
            retriever_scores=selected_scores,
            selected_indices=selected_indices,
        )

    def save_checkpoint(
        self, output: str | Path, metadata: dict[str, Any]
    ) -> Path:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        self.query_encoder.save_pretrained(destination / "query_encoder")
        self.memory_encoder.save_pretrained(destination / "memory_encoder")
        self.retriever_tokenizer.save_pretrained(destination / "retriever_tokenizer")
        self.reader.save_pretrained(destination / "reader")
        self.reader_tokenizer.save_pretrained(destination / "reader_tokenizer")
        document = {
            "architecture": ARCHITECTURE,
            "retriever_model": self.retriever_model_name,
            "reader_model": self.reader_model_name,
            "retriever_dimension": self.retriever_dimension,
            **metadata,
        }
        (destination / "metadata.json").write_text(
            json.dumps(document, indent=2), encoding="utf-8"
        )
        return destination


class EMDR2RuntimeRetriever:
    """Exact local MIPS over the canonical event ledger using learned encoders."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cuda") -> None:
        from transformers import AutoModel, AutoTokenizer

        self.checkpoint = Path(checkpoint)
        self.metadata = json.loads(
            (self.checkpoint / "metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("architecture") != ARCHITECTURE:
            raise ValueError("unsupported EMDR2 checkpoint architecture")
        if not bool(self.metadata.get("accepted_retriever_for_runtime")):
            raise ValueError("EMDR2 retriever did not pass offline acceptance")
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint / "retriever_tokenizer"
        )
        self.query_encoder = AutoModel.from_pretrained(
            self.checkpoint / "query_encoder"
        ).to(self.device).eval()
        self.memory_encoder = AutoModel.from_pretrained(
            self.checkpoint / "memory_encoder"
        ).to(self.device).eval()
        self.records: list[dict[str, Any]] = []
        self.embeddings = torch.empty(
            (0, int(self.metadata["retriever_dimension"])),
            dtype=torch.float32,
            device=self.device,
        )

    def _tokens(self, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
        values = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in values.items()}

    @torch.inference_mode()
    def _encode(self, model: nn.Module, texts: list[str], max_length: int) -> torch.Tensor:
        tokens = self._tokens(texts, max_length)
        output = model(**tokens)
        return _mean_pool(output.last_hidden_state, tokens["attention_mask"]).float()

    @torch.inference_mode()
    def refresh(self, records: list[dict[str, Any]], batch_size: int = 128) -> None:
        self.records = list(records)
        batches = [
            self._encode(
                self.memory_encoder,
                [str(record["content"]) for record in self.records[offset : offset + batch_size]],
                192,
            )
            for offset in range(0, len(self.records), batch_size)
        ]
        self.embeddings = (
            torch.cat(batches, dim=0)
            if batches
            else torch.empty(
                (0, int(self.metadata["retriever_dimension"])),
                dtype=torch.float32,
                device=self.device,
            )
        )

    @torch.inference_mode()
    def add(self, record: dict[str, Any]) -> None:
        vector = self._encode(self.memory_encoder, [str(record["content"])], 192)
        self.records.append(dict(record))
        self.embeddings = torch.cat((self.embeddings, vector), dim=0)

    @torch.inference_mode()
    def retrieve(self, query: str, limit: int = 5) -> list[MemoryHit]:
        if not self.records or limit <= 0:
            return []
        vector = self._encode(self.query_encoder, [query], 192)
        scores = (vector @ self.embeddings.T)[0]
        count = min(int(limit), len(self.records))
        values, indices = torch.topk(scores, k=count)
        hits: list[MemoryHit] = []
        for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
            record = self.records[index]
            hits.append(
                MemoryHit(
                    event_id=str(record["event_id"]),
                    content=str(record["content"]),
                    actor=str(record["actor"]),
                    similarity=float(value),
                    occurred_at=str(record["occurred_at"]),
                    decision_action=record.get("decision_action"),
                )
            )
        return hits
