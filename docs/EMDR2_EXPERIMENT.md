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
- Reranker: MPC-BERT or BERT-base cross-encoder with a binary usefulness head.
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
