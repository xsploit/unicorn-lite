# Unicorn Lite: Experiment Log

This repository is the record of an attempt to build something more interesting
than a Discord wrapper that calls an LLM after every ping. The target is a local,
persistent controller that observes every experience, maintains state, decides
whether speaking is worthwhile, and treats an LLM as an optional writer.

The project is experimental. Results below distinguish measured behavior from
architecture ideas, and failed challengers remain documented instead of being
quietly rewritten as successes.

## 1. Feasibility boundary

The first idea was to reproduce a research-style agent with permanent neural
memory, variable compute, learned action selection, and optional test-time
training. Full In-Place TTT was rejected as the starting controller: its reference
recipes focus on adapting multi-billion-parameter language models over long
contexts. That would spend the 16 GB GPU budget on the writer instead of testing
the more important hypothesis—whether a cheap non-generative system can decide
when speech is useful.

The chosen boundary was:

```text
experience -> local encoder -> recurrent state -> associative memory
           -> local response/compute policy -> optional writer
```

The immutable SQLite ledger remains canonical. Neural state is useful, bounded,
and allowed to be lossy; it is not permitted to silently rewrite history.

## 2. First local core

The first working core combined:

- a local sentence encoder;
- a recurrent PyTorch controller;
- a 96 x 96 surprise-scaled fast-weight associative memory;
- versioned facts and immutable events;
- OBSERVE, cheap-reply, and deep-reply actions;
- optional Ollama and Vercel writers.

On an RTX 5060 Ti 16 GB, a 100-event smoke test processed 96.17 events/second,
allocated 96.7 MiB of CUDA memory at peak, and made zero API calls for background
events. This established that observing everything does not require generating
text for everything.

## 3. Learning when to speak

Chronological Discord history was converted into causal training rows. A response
is positive only when the historical agent actually replied or was explicitly
linked to the message. Features are calculated only from information available at
that point in time to prevent future-response leakage.

The initial learned response gate was selective but had weak conversational
representations. MiniLM, BERT-base, and MPC-BERT were compared as frozen local
encoders. MPC-BERT transferred best because its pretraining explicitly models
speaker and reply structure.

The accepted MPC-BERT gate at the measured 0.28 channel threshold produced:

| Replay | AP | Precision | Recall | F1 | Reply rate |
|---|---:|---:|---:|---:|---:|
| Neuro general, 1,282 turns | 0.919 | 0.963 | 0.767 | 0.854 | 6.4% |

The persistent MiniLM representation core and MPC-BERT speech gate are separate.
Changing the gate encoder therefore does not invalidate stored memory vectors.

## 4. Adaptive recursive computation

The gate was extended with a small recurrent ponder loop. This is not recursive
LLM prompting. The local GRU may reuse its hidden state for one to eight steps and
a learned halt head decides when further computation is not worth its cost.

Audit embeds expose the entire probability path and halt decision. A path may rise
or fall because every recurrent update changes the hidden representation. The
last selected probability is compared with the channel threshold.

## 5. Failure: relevance became repetition

A live test asked the same semantic question twice within roughly 92 seconds. The
first turn scored 0.341 and received a reply. The second scored 0.828 and also
received a reply. Answered-message similarity had made the repeat look highly
relevant without representing that Neuro had already handled it.

Several larger retraining attempts suppressed the repeat but damaged legitimate
questions. They were rejected rather than deployed.

The accepted repair freezes the proven 22-feature gate and trains a four-input
additive-logit calibrator on 21 repeat-negative pairs balanced by 21 legitimate
follow-up-positive pairs. To prevent unrelated score drift, deployment activates
only one learned weight:

```text
recent answered message >= 0.90 similarity and <= 10 minutes old: -3.195 logits
```

On the 1,282-turn causal replay, only eight decisions change. All eight are
repeat/spam loops: duplicated commands, bare mentions, emoji, or mass mentions.
The live failure trace becomes REPLY then SILENCE (0.341 then 0.164), and the
separate five-message humanity/spam trace remains 5/5 correct.

Historical aggregate F1 falls to 0.802 because the log labels those eight actual
historical bot replies as positives. This is an example of an evaluation metric
penalizing the intended behavior change; the changed-turn audit is the stronger
acceptance test here.

## 6. Current architecture

```text
Discord event
   |-- immutable event ledger
   |-- MiniLM persistent representation and associative memory
   |-- MPC-BERT conversation representation
   v
local recurrent ponder gate (1..8 steps)
   v
learned answered-message cooldown
   |-- SILENCE: no writer call
   `-- REPLY: Vercel or local writer receives selected context
```

The writer cannot alter routing, canonical memory, or the response threshold.

## 7. Next experiment: EMDR2-inspired memory retrieval

[_EMDR2_](https://arxiv.org/abs/2106.05346) trains retrieval from downstream answer usefulness rather than treating
embedding similarity as the final authority. Its exact reader-retriever gradient
path cannot be reproduced through a black-box hosted writer, but the useful idea
can be tested locally:

1. MiniLM retrieves about 20 candidate memories.
2. A small local cross-encoder reranks `(message, memory)` pairs.
3. The top five memories reach the writer only after the speech gate selects
   REPLY.
4. Historical `(message, prior memory, Neuro reply)` triples provide offline
   answer-aware supervision.
5. The challenger is accepted only if held-out reply retrieval and live memory
   attribution improve without raising writer-call rate.

Natural Questions, TriviaQA, and WebQuestions may warm-start factual passage
retrieval. They are not substitutes for Discord data when learning callbacks,
relationships, conversation state, or when to remain silent.

## Reproducibility and privacy boundary

Source, tests, training utilities, and aggregate measurements are published. Raw
Discord histories, the live SQLite database, credentials, model weights, runtime
logs, and the conversion virtual environment are not. They are not required to
inspect the implementation or run the synthetic/unit tests.
