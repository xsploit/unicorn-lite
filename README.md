# Unicorn Lite

Read the measured successes, rejected challengers, live repeat failure, and next
research step in [the experiment journey](docs/JOURNEY.md).

This is a buildable approximation of the architecture we discussed: a persistent
local controller that can use an LLM, rather than an LLM pretending to be the
entire agent.

```text
every experience
      |
local sentence encoder
      |
persistent recurrent state + fast-weight associative memory
      |
versioned facts + immutable event memory
      |
cost-aware policy: observe / cheap reply / deep reply
      |
optional local Ollama writer or explicitly selected Vercel writer
```

Most messages do **not** call an LLM. They still enter the event log, update the
neural controller, change its associative memory, and may revise versioned facts.
The controller's state is written to SQLite after every event and restored after a
restart.

## What is real in this prototype

- A local pretrained encoder turns experiences into 384-dimensional tensors.
- A PyTorch recurrent controller carries active state between experiences.
- A persistent 96 x 96 fast-weight matrix learns associations online with a
  surprise-scaled delta rule. This updates during inference without gradients.
- Raw events remain immutable. Corrections create a new fact version and close the
  old version.
- An accepted local learned policy chooses silence or speech for every message.
  A ping is one learned feature, never a hardcoded obligation.
- The writer is replaceable and has no authority over memory or routing.
- Replay training can tune the recurrent controller for next-experience prediction
  and response-selection labels.

The development runtime has a locally trained controller checkpoint at
`data/neuro-controller.pt`; model weights and private replay data are deliberately
not published. CLI, chat, and Discord modes load that path by default;
`UNICORN_CHECKPOINT` or `--checkpoint` can select another checkpoint.

## What is deliberately not claimed

- This is not ByteDance's full In-Place TTT recipe.
- It does not continuously retrain a large language model.
- CLI demos retain an inspectable cold-start fallback. Discord mode refuses to
  start unless its learned checkpoint passed offline acceptance.
- The local encoder represents text; it is not yet a multimodal world model.
- Fast-weight memory is bounded active neural memory. The immutable database is the
  permanent source of truth.

That boundary is intentional: it gives us something measurable on a 16 GB GPU
without disguising an API loop as a novel mind.

## Measured on the RTX 5060 Ti 16 GB

The completed local smoke benchmark processed 100 events through the real
MiniLM encoder, recurrent controller, fast-weight update, vector retrieval, and
SQLite commit:

| Measurement | Result |
|---|---:|
| Sustained processing | 96.17 events/second |
| Peak allocated CUDA memory | 96.7 MiB |
| Active fast-weight memory | 36 KiB |
| API calls for 100 background events | 0 |
| Unprompted reply decisions | 0 |

This leaves almost the entire 16 GB card available for a future local writer or a
larger representation model. It also proves that processing every Discord message
does not imply pinging an LLM for every Discord message.

The local end-to-end writer path uses `neuro-gemma4-rp:latest` through Ollama.
On the RTX 5060 Ti it runs 100% on GPU; after cold loading it measured about
92 output tokens/second. Active Discord startup warms it and asks Ollama to keep
it resident. Vercel remains an explicitly selected alternative.

## Run it

Copy `.env.example` to `.env` and provide only the services you intend to use.
`UNICORN_ENV_FILE` may point to another read-only environment file if desired.

Use Python 3.10 because that is the installed environment with working CUDA.

```powershell
Set-Location "C:\path\to\unicorn-lite"
py -3.10 -m unicorn_lite doctor
py -3.10 -m unicorn_lite doctor --verify-discord
py -3.10 -m unicorn_lite demo --encoder hash --device cuda
```

The hash demo requires no download. For the actual local neural text encoder:

```powershell
py -3.10 -m unicorn_lite demo --encoder local --device cuda
```

`all-MiniLM-L6-v2` downloads once and then runs locally. The database defaults to
`./data/unicorn.db`.

## Import the Neuro persona and enable a writer

The Neuro prompt can be extracted from the WebWaifu local-transfer backup without
copying any provider secrets:

```powershell
py -3.10 -m unicorn_lite import-persona
```

Ollama is the default writer. It is still opt-in for CLI/chat:

```powershell
py -3.10 -m unicorn_lite chat --encoder local --allow-writer --writer ollama
```

To explicitly select Vercel instead:

Set `AI_GATEWAY_API_KEY`, then explicitly allow the demo to call the API:

```powershell
$env:AI_GATEWAY_API_KEY = "your-key"
py -3.10 -m unicorn_lite demo --encoder local --allow-writer --writer vercel
```

Normal replies route to `deepseek/deepseek-v4-flash`; long research, debugging, or
architecture requests route to `deepseek/deepseek-v4-pro`.

If the gateway is unavailable or returns a billing error, the event and neural
state are still committed. The CLI reports the writer failure and the local agent
continues instead of losing its experience or crashing.

Interactive mode:

```powershell
py -3.10 -m unicorn_lite chat --encoder local
```

Interactive chat and one-shot ingest are local-only by default. Add
`--allow-writer` only when you explicitly want selected replies rendered. The
controller, memory, scraper, labels, trainer, shadow listener, and response gate
never require a generative model or external API.

## Discord mode

The token is loaded from `DISCORD_BOT_TOKEN`. Shadow mode is the default:

```powershell
py -3.10 -m unicorn_lite discord --encoder local --mode shadow
```

Every non-bot message is processed, but shadow mode never calls Vercel or sends a
Discord message. A bounded connection test can close itself automatically:

```powershell
py -3.10 -m unicorn_lite discord --encoder local --mode shadow --run-seconds 20
```

Only `--mode active` permits writer calls and replies. Active mode is still
DM/mention-only by default. `--allow-unsolicited` enables learned initiative, and
repeated `--unsolicited-channel-id` arguments constrain it to named channel IDs.
The development launcher permits learned unsolicited participation only in an
explicit channel allowlist. Its currently measured general-chat threshold is
0.28; thresholds are checkpoint- and dataset-specific and should not be copied to
a new server without replay evaluation. Bot-authored messages can update context
but can never trigger a reply, preventing bot loops. Selected writer calls receive
the latest ten speaker-labelled channel messages for local conversational
continuity.

The active runtime can use either writer. The persistent launcher currently uses
Vercel AI Gateway; the learned local controller still decides whether any writer
call is allowed:

```powershell
py -3.10 -m unicorn_lite discord --encoder local --mode active --writer vercel
```

For a persistent Windows runtime, copy the published example and supply your
channel IDs:

```powershell
Copy-Item .\neuro-runtime.example.ps1 .\neuro-runtime.ps1
.\neuro-runtime.ps1 Start
.\neuro-runtime.ps1 Status
.\neuro-runtime.ps1 Stop
```

It prevents duplicate launches, runs hidden, and keeps separate stdout/stderr logs
in this project directory.

### Backfill chronological training data

Inventory what the configured bot can read, then collect history without sending
messages or calling an LLM:

```powershell
py -3.10 -m unicorn_lite discord-inventory
py -3.10 -m unicorn_lite discord-backfill --limit-per-channel 2000
```

Use `--guild-id` or repeat `--channel-id` to narrow the source. A limit of `0`
requests all available history. The collector writes globally chronological JSONL,
preserves reply and mention relationships, includes agent messages for response
labeling, and deduplicates by Discord message ID on later runs.

Conservative labels can then be derived for a specific historical agent. Only an
explicit Discord reply or an explicit author mention becomes a positive response:

```powershell
py -3.10 -m unicorn_lite discord-labels .\data\discord-agent-history.jsonl `
  --target-author-id YOUR_AGENT_ID `
  --output .\data\discord-lewwa-policy.jsonl
```

The output retains timestamps, source message IDs, directness, response delay,
the actual expected reply, and the label source. This makes uncertain cases
auditable before they are admitted to controller training.

Train the first local learned reply/silence gate with a chronological 80/20 split:

```powershell
py -3.10 -m unicorn_lite train-discord-gate `
  .\data\discord-lewwa-policy.jsonl `
  --output .\data\discord-response-gate.pt
```

The trainer uses the local sentence encoder, causal interaction features, and a
small recurrent policy. It uses a chronological train/calibration/test split;
the threshold is selected on calibration and the latest 15 percent remains
untouched until the final report. The accepted checkpoint is a learned-only gate:
a ping is an input feature, never an obligation to reply.

The original selective checkpoint, `data/uni-selective-response-gate.pt`, was
trained on Uni's history plus bots-general context. Uni was selected as the
teacher because its historical behavior was meaningfully selective; Neuro's
history answered almost every direct request and would mostly reproduce a
ping-to-reply rule. Bots-general supplies crowded multiparty context and negative
examples, not persona. Decorative punctuation is removed from the controller's
semantic input, and question form is derived from words rather than punctuation
count, so strings such as `????` or `>>>` cannot inflate the reply score. On 328
held-out later messages the selective gate reached 0.901 average precision, 0.828
precision, 0.857 recall, and 0.842 F1 while selecting 26.5 percent of messages.
The direct-only baseline reached 0.855 average
precision, 0.905 precision, 0.798 recall, and 0.848 F1 while selecting 22.6
percent. The checkpoint passed the `initiative_balanced` acceptance profile: at
least 98 percent of the direct baseline's ranking and F1, higher recall than the
direct-only rule, and no more than a 30 percent writer-call rate.

The live successor, `data/uni-transfer-response-gate.pt`, expands the structural
input from 16 to 22 features. The added inputs are causal retrieval signals:
maximum and mean top-memory similarity, near-duplicate count, exact-duplicate
status, maximum similarity to an event that received a reply, and a recent
answered-exact-repeat flag. Old 16-feature checkpoints remain loadable.

It is initialized on 6,000 balanced natural multi-party examples from
`ishiki-labs/multi-party-dialogue`, then fine-tuned and recalibrated on the same
chronological Discord set. On the 328-message Discord holdout it reached 0.889
precision, 0.857 recall, 0.873 F1, and a 24.7 percent reply rate. On the separate
1,282-message historical Neuro bots-general replay it reached 0.927 precision,
0.738 recall, and a 6.4 percent reply rate versus the observed 8.0 percent.

### Adaptive-depth local controller

`data/ponder-response-gate.pt` is the experimental live successor. It uses the
same local sentence embedding, causal features, and persistent per-channel state,
but can reuse its small GRU controller for one to eight updates before choosing
reply or silence. A learned halt head chooses the depth; training supervises every
depth and charges an expected-compute penalty. This is recursive local controller
compute, not repeated LLM prompting. The Vercel writer is still unreachable until
the final gate selects speech.

Train it from the accepted one-step checkpoint:

```powershell
py -3.10 -m unicorn_lite train-discord-gate `
  .\data\discord-uni-memory-policy.jsonl `
  --output .\data\ponder-response-gate.pt `
  --architecture ponder `
  --max-ponder-steps 8 `
  --compute-penalty 0.015 `
  --stability-penalty 0.01 `
  --init-checkpoint .\data\uni-transfer-response-gate.pt
```

On the untouched 328-message Discord test slice it reached 0.895 average
precision, 0.908 precision, 0.821 recall, and 0.863 F1 at a 23.2 percent reply
rate. It used 1.13 controller steps on average: 319 messages halted at step one
and nine uncertain cases used additional computation. The previous one-step gate
reached 0.891 average precision, 0.889 precision, 0.857 recall, and 0.873 F1 at a
24.7 percent reply rate. This is a precision/compute experiment, not a universal
quality win: recall and F1 are slightly lower.

Depth ablation across the complete 2,181-message Discord policy set found that
adaptive depth beat fixed one-step and forced eight-step average precision
(0.8893 versus 0.8878 and 0.8880). The gain was larger on implicit turns
(0.6550 versus 0.6430 and 0.6224). Forced extra computation was therefore worse;
the learned halt decision is the useful part. Decision-audit embeds expose the
selected probability and step count; the attached schema-v2 JSON retains the
halt probability and complete reply-probability path.

The external tests remain a warning. On Ishiki and When2Speak, the controller
spent roughly 3.1 steps but severely over-replied because their conversation and
label distributions differ from Discord. Recursive depth detects unfamiliarity;
it does not create domain calibration by itself.

### MPC-BERT conversation-encoder experiment

The first representation challenger is now real and local. It changes the
encoder underneath the persistent controller; it does not turn the controller
into an LLM loop:

```text
every Discord event
        |
frozen local conversation encoder
(MiniLM, BERT-base, or MPC-BERT)
        |
persistent non-LLM controller
  + active recurrent state
  + temporal external memory
  + response and compute policy
        |
   SILENCE or invoke writer
        |
Vercel LLM only after REPLY
```

[MPC-BERT](https://aclanthology.org/2021.acl-long.285/) is a BERT-base-sized
encoder pretrained for multi-party conversation: speaker identity, reply-to
structure, addressee recognition, and response selection. Its official release
is a legacy TensorFlow checkpoint. `scripts/export_mpcbert_tf.py` exports only
the 200 inference tensors in an isolated TensorFlow environment, and
`scripts/convert_mpcbert_checkpoint.py` converts them plus the learned
16-speaker embedding table to `data/mpcbert-encoder.pt`. The active runtime never
imports TensorFlow. `google-bert/bert-base-uncased` is the equal-width 110M
parameter control.

Both challengers use utterance-level `[CLS]` markers and the final utterance
representation. MPC-BERT additionally applies its learned speaker embeddings to
`AGENT`, `CURRENT_USER`, `OTHER_USER`, and `OTHER_BOT` turns. Both encoders are
frozen; only the small recurrent reply gate is trained on the 2,181 chronological
Discord examples.

Held-out and transfer results:

| Encoder | Discord AP | Precision | Recall | F1 | Reply rate | Neuro-general AP | Neuro-general F1 | Neuro-general reply rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM live | 0.895 | **0.908** | 0.821 | 0.863 | **23.2%** | 0.868 | 0.840 | **6.6%** |
| BERT-base | **0.920** | 0.794 | 0.917 | 0.851 | 29.6% | 0.903 | 0.830 | 8.5% |
| MPC-BERT | 0.896 | 0.868 | **0.940** | **0.903** | 27.7% | **0.908** | **0.849** | 8.0% |

The attached five-message spam trace was also replayed with the desired labels:
reply to the real humanity question, then stay silent for the checkmark, `piss`,
and both troll-emoji turns. MPC-BERT got 4/5, while BERT-base and MiniLM each got
2/5. MPC-BERT correctly silenced the repeated explicitly pinged troll emoji at
0.147, below its 0.170 threshold; its miss was the unpinged troll emoji. This is a
small targeted stress test, not a general benchmark, and its exact inputs are in
`data/live-trace-microeval.jsonl`.

Measured on the RTX 5060 Ti, MPC-BERT costs about 9.7 ms per live-style message,
uses about 438 MiB peak CUDA allocation for single-message inference, and encodes
about 183 events/second in batches of 64. BERT-base was 10.3 ms and MiniLM was
4.8 ms. MPC-BERT is therefore operationally feasible, but is deliberately not
the live default yet: it trades some precision and more writer calls for recall,
general-chat F1, and better behavior on the targeted repeated-ping trace.

### Matched hard-negative experiment

The useful negative data is not more random background chat; the base set already
contains 1,817 silent turns. Its blind spot was *direct but low-value* input. A
training-only selector now imports concise, pre-cutoff direct silences such as
empty mentions, greetings, acknowledgements, insults, and punctuation while
excluding unanswered questions and requests. A counterfactual builder also takes
real pre-cutoff emoji-only turns and changes only the addressee edge, producing
examples such as `@Neuro <emoji>`. This teaches that a mention is context, not an
automatic command to speak.

Hard negatives must be balanced with matched direct positive anchors. Training
only on extra silence made the controller reject the real humanity question as
well. The selected model uses 18 natural direct negatives, 16 counterfactual
mention-plus-emoji negatives, and 34 concise direct positive questions/requests.
The source manifests remain inspectable in
`data/mpcbert-hard-negative-manifest.json` and
`data/counterfactual-direct-emoji-negatives.jsonl`.

At its globally calibrated checkpoint threshold of 0.34 the model was too quiet
for Neuro-general. At a measured channel threshold of 0.28 it achieved the
following replay result without a writer call:

| Evaluation | AP | Precision | Recall | F1 | Reply rate |
|---|---:|---:|---:|---:|---:|
| Held-out Discord | 0.922 | 0.894 | 0.905 | 0.899 | 25.9% |
| Neuro-general, 1,282 turns at 0.28 | 0.919 | **0.963** | 0.767 | **0.854** | **6.4%** |
| Current live MiniLM Neuro-general | 0.868 | 0.929 | 0.767 | 0.840 | 6.6% |

On the five-turn failure trace, threshold 0.28 replies only to the real humanity
question (0.285) and silences the checkmark (0.241), `piss` (0.061), unpinged
troll emoji (0.268), and pinged troll emoji (0.252). This is the first challenger
that gets all five desired outcomes while also improving the full replay's
precision and F1. The full reports are
`data/mpcbert-counterfactual-balanced-neuro-general-threshold-028-eval.json` and
`data/mpcbert-counterfactual-balanced-effective-direct-threshold-028-microeval.json`.

Live metadata now defines `direct` consistently with training: DM, actual mention,
lexical address, or reply-to-agent. The MPC-BERT challenger is deployed for the
speech gate, with a measured `0.28` override in Neuro-general and the checkpoint's
globally calibrated `0.34` threshold elsewhere.

### Learned answered-message cooldown

The accepted gate originally answered a semantically identical question twice
within 92 seconds: the first score was 0.341 and the second rose to 0.828 because
retrieval similarity was treated as relevance rather than evidence that Neuro had
already answered it. The runtime now retrieves recent answered messages separately
and exposes their similarity and age to a tiny additive-logit calibrator. The
deployed calibrator was trained on 21 repeat-negative pairs and 21 legitimate
follow-up-positive pairs while the accepted MPC-BERT gate remained frozen.

Only the learned `recent_answered_near_duplicate` weight is active (`-3.195`), so
the checkpoint is mathematically identical to the accepted gate unless an answered
message is at least 0.90 similar and no more than 10 minutes old. On the 1,282-turn
Neuro-general replay it changes exactly eight decisions; every changed turn is a
repeat/spam loop such as repeated `!bot`, repeated bare mentions, or duplicated
mass mentions. The held-out real repeat trace changes from REPLY/REPLY to
REPLY/SILENCE, while the five-message humanity/spam trace remains 5/5 correct.
The old aggregate F1 falls from 0.854 to 0.802 because those eight historical bot
replies are labeled positive even though suppressing them is the intended behavior.

Train and audit the isolated calibrator:

```powershell
$env:PYTHONPATH = '.'
py -3.10 .\scripts\train_refractory_calibrator.py `
  .\data\discord-uni-refractory-policy.jsonl `
  .\data\refractory-repeat-followup-pairs.jsonl `
  .\data\mpcbert-counterfactual-balanced-ponder-response-gate.pt `
  --output .\data\mpcbert-refractory-near-only-ponder-response-gate.pt `
  --encoder mpc-bert --device cuda --near-only
```

The Discord runtime supports a dedicated `--gate-encoder`. The deployed layout
keeps MiniLM as the persistent core and vector-memory encoder, preserving its
trained checkpoint and stored neural state, while MPC-BERT independently encodes
the conversation for the reply/silence gate. The extra local pass is preferable
to changing the meaning and dimension of the existing memory vectors.

Train or inspect the challengers:

```powershell
py -3.10 -m unicorn_lite train-discord-gate `
  .\data\discord-uni-memory-policy.jsonl `
  --output .\data\mpcbert-ponder-response-gate.pt `
  --encoder mpc-bert --architecture ponder --epochs 60

py -3.10 -m unicorn_lite train-discord-gate `
  .\data\discord-uni-memory-policy.jsonl `
  --output .\data\bert-base-ponder-response-gate.pt `
  --encoder bert-base --architecture ponder --epochs 60

py -3.10 -m unicorn_lite evaluate-discord-gate `
  .\data\live-trace-microeval.jsonl `
  .\data\mpcbert-ponder-response-gate.pt `
  --encoder mpc-bert --include-examples

py -3.10 -m unicorn_lite evaluate-discord-gate `
  .\data\discord-neuro-general-memory-policy.jsonl `
  .\data\mpcbert-counterfactual-balanced-ponder-response-gate.pt `
  --encoder mpc-bert --threshold 0.28 --include-examples
```

New checkpoints store an encoder identity and refuse an equal-dimensional but
incorrect encoder. The event ledger also ignores embeddings from other vector
dimensions during retrieval, so 384- and 768-dimensional experiments can share
the immutable database safely.

The architecture description observed in the Lewwa discussion suggests a later,
separate speech experiment: sample candidate replies from a latent sentence
decoder, score candidates using learned fuzzy/neuro-symbolic satisfaction and
cost signals, then reject or stay silent. That should not replace the proven
writer until candidate quality is measured. Text VAEs are feasible on this GPU,
but fluent open-domain Discord speech is a much harder target than response
selection and is prone to bland generation or latent collapse.

### What EMDR2 contributes

[EMDR2](https://arxiv.org/abs/2106.05346) is not a response/silence controller.
It jointly improves a retriever and a multi-document answer reader by treating
the relevant document set as a latent variable and approximating the otherwise
intractable training objective with expectation maximization. Its useful lesson
for this project is that memory retrieval should eventually be trained from
downstream answer usefulness rather than cosine similarity alone.

A feasible adaptation on this machine is a separate learned memory reranker:
MiniLM retrieves roughly 20 Discord-memory candidates, a small local cross-encoder
reranks them, and only the best five are sent to the writer. Historical
message-to-Neuro-reply pairs can provide answer-aware teacher labels. Natural
Questions, TriviaQA, and WebQuestions may warm-start factual passage retrieval,
but they should not train the speech gate or social-memory policy. Reproducing
EMDR2 literally would require differentiating through the reader; that is not
possible with the current black-box Vercel writer and is unnecessary for the
first reranker experiment.

External data can be converted and evaluated without asking an LLM for JSON:

```powershell
py -3.10 -m unicorn_lite import-turntaking-data ishiki `
  .\data\external\ami\test\test_samples.jsonl `
  .\data\external\friends\test\test_samples.jsonl `
  --output .\data\external\ishiki-test-policy.jsonl

py -3.10 -m unicorn_lite evaluate-discord-gate `
  .\data\external\ishiki-test-policy.jsonl `
  .\data\uni-transfer-response-gate.pt `
  --output .\data\external\ishiki-transfer-gate-eval.json
```

`duke-trust-lab/When2Speak` is also supported with format `when2speak`. Treat its
synthetic 87-percent-silent distribution as an abstention stress test or
pretraining source, not as a drop-in replacement for real Discord labels.

## Train the controller from a replay

JSONL records need `content` and may include an `action` label:

```json
{"content":"background chatter","action":"OBSERVE"}
{"content":"@agent can you help?","action":"REPLY_FLASH"}
{"content":"research and compare these architectures","action":"REPLY_PRO"}
```

Train and load the result:

```powershell
py -3.10 -m unicorn_lite train .\replay.jsonl --output .\data\controller.pt
py -3.10 -m unicorn_lite chat --checkpoint .\data\controller.pt
```

The cold-start policy is not authoritative in Discord mode. The live launcher
explicitly loads `data/uni-transfer-response-gate.pt`, persists its per-channel
hidden state in SQLite, and refuses checkpoints whose offline acceptance flag is
false.

The existing Neuro Q/A corpus can train the representation transition without
pretending that independent Q/A rows are a chronological life history:

```powershell
py -3.10 -m unicorn_lite train-neuro --epochs 12 --output .\data\neuro-controller.pt
py -3.10 -m unicorn_lite chat --checkpoint .\data\neuro-controller.pt
```

This teaches the controller to map a user experience toward the expected reply
representation and gives its response/compute heads supervised direct-request
examples. It reports validation metrics before and after training. It cannot
teach when to remain silent or how memories evolve across time because each row
is an independent user/assistant pair; chronological Discord shadow logs are the
correct next dataset for those behaviors.

The first 12-epoch run on 922 training pairs produced these held-out results on
48 validation pairs:

| Metric | Random controller | Trained controller |
|---|---:|---:|
| Paired next-reply cosine | 0.000 | 0.335 |
| Correct reply retrieval, top 1 of 48 | 4.2% | 27.1% |
| Correct reply retrieval, top 5 of 48 | 10.4% | 47.9% |
| Mean reciprocal rank | 0.110 | 0.395 |

The routing labels also reached 100% on this small validation split, but those
rows are all direct Q/A interactions. The result is not evidence that the neural
head knows when to remain silent, so the cost-aware cold-start policy remains the
runtime authority.

## Where In-Place TTT fits later

ByteDance's implementation makes selected LLM projection weights adapt as tokens
are read. That is useful as a long-context writer/reader backend, but its reference
training recipes target multi-billion-parameter models, long contexts, distributed
training, and tens of billions of training tokens. It is not the economical first
controller for this GPU.

A later experiment can replace the Vercel writer with a quantized local writer and
an In-Place-TTT adapter. The local controller and canonical memory should remain
outside it so document resets, model swaps, or writer failures do not erase the
agent.

## Test

```powershell
py -3.10 -m unittest discover -s tests -v
```
