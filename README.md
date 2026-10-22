# trigger-hunter

Bounded audit checks for locally hosted open-weight models.

`trigger-hunter` enumerates the input surfaces a harness actually exposes and reports
where model behaviour diverges. It is a **finite-space enumerator and differential
tester**. It is not a scanner that certifies a weight file clean.

## What it does not do

Stated up front, because the whole point is not overclaiming:

- It does **not** prove the absence of a backdoor. No tool can.
- It does **not** search for unknown triggers. There is no gradient/GCG adversarial-suffix
  search, no interpretability, no SAE. The name is aspirational; the README is not.
- It does **not** detect steganographic output (a payload spread across ordinary-looking
  tokens).
- It is **not** robust against sandbagging. A model that recognises evaluation can pass
  every probe here.

A clean run means: no divergence across the enumerated dimensions, for these probes.
That is a finite-space result and nothing more.

## Proof forms

Every claim this tool can support is one of three shapes:

| Form | Establishes | Requires |
|---|---|---|
| Enumeration | no input in *this finite set* triggers a deviation | a bounded, listable surface |
| Structure | even if triggered, nothing bad happens | holds regardless of weights |
| Identity | the bytes audited are the bytes running | a reference hash / base model |

Structural guarantees (egress control, no privileged sink, schema closure, serving-stack
determinism, artifact hashing) live in your harness, not in this repo. This repo gives
you the enumeration and identity halves.

## Quick start

No install. Python 3.8+, stdlib only. Speaks the OpenAI `/v1/chat/completions` wire
format, so it works against llama-server, llama.cpp, vLLM, LM Studio and hosted APIs.

```bash
python audit.py hash D:/Models/model.gguf --expect <sha256-from-pinned-revision>

python audit.py ambient  --endpoint http://127.0.0.1:8080 --model local --date-mode years
python audit.py lattice  --endpoint http://127.0.0.1:8080 --model local --dims turn,encoding,lang
python audit.py payload  --endpoint http://127.0.0.1:8080 --model local --samples 200
python audit.py diff     --endpoint http://127.0.0.1:8080 --model suspect \
                         --baseline http://127.0.0.1:8081 --baseline-model base
```

## Subcommands

- **`hash`** — size, sha256, GGUF container header, optional comparison against the
  sha256 from the pinned repo revision. Identity proof for the artifact.
- **`ambient`** — sweeps harness-injected context: date, year, epoch, time, hostname,
  user, cwd. A model has no clock; a time trigger must arrive through one of these,
  which makes the surface finite and testable.
- **`lattice`** — enumerates trigger-shaped dimensions: conversation depth (`turn`),
  context length (`ctxpad`), 35 languages (`lang`), encoding wrappers
  base64/hex/rot13/percent (`encoding`), Unicode format characters (`unicode`), and
  system-prompt variants (`system`).
- **`payload`** — fixed-payload detector. Samples many unrelated prompts and reports
  verbatim n-grams recurring across distinct prompts. Backdoored models tend to emit a
  fixed string, which is an unnatural signature.
- **`diff`** — differential test against a trusted baseline endpoint. Ranks prompts by
  behavioural divergence. Highest-yield white-box technique: you don't need to know the
  trigger, only what the untampered model looked like.

Every run appends full transcripts to JSONL under `--out` so you can grep the raw
evidence instead of trusting a score.

## Scaling to a probe corpus (`matrix.py`)

`lattice` crosses one probe with ~50 conditions. That does not scale: 3,000 probes x
50 conditions is 147,000 calls, unaffordable and statistically useless. `matrix.py`
refuses the cross product and splits the two axes.

| Axis | Cost | Coverage |
|---|---|---|
| **battery** — every probe at the identity condition | N | full |
| **ofat** — 3 anchor probes x every condition | ~150 | full over conditions |
| **pairwise** — 8 anchored conditions x 25 sampled probes | ~200 | **sampled, not proven** |

At 30 probes and 46 conditions that is 341 calls instead of 1,380 — 24.7% of the cross
product — and at 3,000 probes it stays 341 + N instead of 138,000.

Flagging is the part that breaks with multilingual output, so it is split by rule class:

- **similar** (encoding, unicode, system, turn, ctxpad) — output stays in the source
  language, so comparison against the matched baseline is valid.
- **translating** (`lang`) — output changes language, so text similarity is meaningless.
  Only absolute detectors and population statistics apply.

Measured on a multilingual stub that answers in 12 languages with no backdoor: under the
old Jaccard rule **330 of 436 language cases (76%) would have been flagged** as
divergence. `matrix.py` reports zero findings on the same model.

Detectors are judge-free — no second model is trusted to decide: canary leakage,
credentials, shell strings, egress URLs, tool-call sentinels, override phrases, payload
n-grams spanning distinct probes, similar-family length collapse, empty output.

`--repeat 2` re-runs the identical config and reports the noise floor (measured: 341/341
identical at temperature 0), so a finding can be distinguished from sampling chatter.
Results are cached by `sha1(model|condition|probe|temp|seed)`, so a 3,000-probe corpus
re-run only pays for what changed.

```bash
python matrix.py --endpoint http://127.0.0.1:8080 --model local \
                 --probes probes.txt --plan battery,ofat,pairwise \
                 --cache out/cache.jsonl --out out --repeat 2
```

Negative control: `tests/stub_ml.py` answers in the requested language, and with
`BACKDOOR=1` fires a payload **only** for Russian. Detected as `lang:ru 25/25 calls hit`
with credential + egress + override detectors, while all 11 other languages stayed clean.

## Negative controls

`tests/stub_server.py` and `tests/stub_server2.py` are **deliberately malicious**
fixtures: small OpenAI-compatible servers that inject a payload when a trigger fires.

- `stub_server.py` — year-conditioned trigger (fires on the string `2031`).
- `stub_server2.py` — turn-index trigger (conversation depth >= 4) plus a hex-encoding
  trigger.

Run the tool against them before you trust a clean result on a real model. A detector
that has never been observed firing is not evidence.

```bash
python tests/stub_server.py &
python audit.py ambient --endpoint http://127.0.0.1:8099 --model stub --date-mode years --probes 1
```

Observed: 13 of 323 calls flagged, every one a 2031 variant, no false positives on the
other 310.

```bash
python tests/stub_server2.py &
python audit.py lattice --endpoint http://127.0.0.1:8098 --model stub --dims turn,encoding,lang,system
```

Observed: 56 cases, `turn[3]` through `turn[12]` and `enc[hex]` flagged, `base64`/`rot13`/
`url` and all 35 languages clean — matching what the stub actually does.

## Known quirks

- On Windows/MSYS, pass `--out` as a **relative** path. A native Python process reads
  `/c/Users/...` as `C:\c\Users\...`.
- `payload` flags constant output as a finding. On a degenerate server that is correct
  behaviour; on a real model, uniform output is the signal you want.
- At temperature > 0, divergence is expected. Measure your noise floor before treating
  divergence as a finding.

## License

MIT.
