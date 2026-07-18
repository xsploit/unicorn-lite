# EMDR2-Inspired Memory Reranker

Research source: [End-to-End Training of Multi-Document Reader and Retriever for
Open-Domain Question Answering](https://arxiv.org/abs/2106.05346).

## Hypothesis

Cosine-nearest memories are not necessarily the memories that improve a reply.
A small local reranker trained from downstream reply evidence should select more
useful context without increasing generative-model calls.

## Scope

This is a memory experiment, not a replacement for the response gate. Silence is
still decided before the writer runs.

## Proposed training record

```json
{
  "query_event": "current Discord message",
  "candidate_memory": "a causally prior message or fact",
  "target_reply": "the historical Neuro reply",
  "useful": true,
  "provenance": ["query event id", "memory event id", "reply event id"]
}
```

Negatives should mix random prior memories, hard cosine-nearest distractors,
near-duplicate answered messages, and memories from the correct conversation that
do not support the target reply.

## First model

- Candidate generator: existing local MiniLM retrieval, top 20.
- Reranker: a small PyTorch selector over frozen MiniLM query/memory vectors.
- Training-only reader: predicts the historical reply embedding so downstream
  answer loss reaches the selector; it is discarded for runtime ranking.
- Deployment: return top 5 with calibrated scores and provenance.
- Writer: unchanged Vercel or local backend.
- GPU target: RTX 5060 Ti 16 GB, mixed precision, short Discord contexts.

## Evaluation

- Recall of known supporting memories at 1 and 5.
- Mean reciprocal rank.
- Reply factual/persona consistency on a frozen held-out set.
- Duplicate-answer rate.
- Writer-call rate and end-to-end latency.
- Attribution audit: which memories changed the rendered answer.

The experiment is rejected if it improves a proxy retrieval metric while making
live replies more repetitive or increasing unsupported personal claims.

## Current status

An all-reply first attempt failed its held-out acceptance test. Adding an
answer-aware latent-memory posterior and filtering for genuinely memory-dependent
interactions produced stable gains across three seeds. The checkpoint remains
opt-in and affects only memory ordering after the response gate has already chosen
to speak. A transactionally backed-up live-database preflight confirmed zero gate
decision/confidence drift, no writer call, changed memory ordering, and about
1.02 ms isolated selector latency before the accepted seed-7 checkpoint was
promoted.

Build and train locally:

```powershell
$env:PYTHONPATH = '.'
py -3.10 .\scripts\build_memory_reranker_dataset.py `
  .\data\discord-history.jsonl `
  --target-author-id YOUR_AGENT_ID `
  --output .\data\memory-reranker-groups.jsonl

py -3.10 .\scripts\train_memory_reranker.py `
  .\data\memory-reranker-groups.jsonl `
  --output .\data\answer-aware-memory-reranker.pt `
  --encoder local --device cuda
```

An accepted checkpoint can be evaluated in a Discord runtime with
`--memory-reranker-checkpoint`. It cannot influence the response gate: the
reranker executes only after the decision action begins with `REPLY`.

Preflight against a safe SQLite backup before promotion:

```powershell
py -3.10 .\scripts\preflight_memory_reranker.py `
  .\data\live.db .\data\answer-aware-memory-reranker.pt `
  --core-checkpoint .\data\controller.pt --encoder local --device cuda
```
