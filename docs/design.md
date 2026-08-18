# System design

## Goal

Minimize the effort between speaking and obtaining text the user is willing to send, while making
latency and personalization behavior measurable rather than implicit.

## Request path

1. The browser sends an utterance as an upload or WebSocket byte chunks.
2. `FasterWhisperTranscriber` runs local VAD and greedy ASR.
3. `ProfileStore` retrieves recent user corrections and channel preferences.
4. `LinUCBBandit` chooses verbatim, balanced, or polished rewriting.
5. A `TextPolisher` applies the deterministic baseline or learned seq2seq model.
6. The API returns the text, chosen mode, and stage timings with an interaction ID.
7. Accept/edit feedback updates both the local profile and contextual-bandit parameters.

## Why a contextual bandit?

Rewrite strength is not a fixed user preference. Short chat messages often need less intervention
than emails, and disfluent dictation often needs more. LinUCB is small enough to inspect, cheap to
update after every interaction, and able to condition its action on channel, length, and measured
disfluency. It is a more honest prototype than claiming to run heavyweight RL from sparse feedback.

## Failure behavior

- The API starts without ASR by default and reports `503` clearly on audio requests.
- Optional model imports are lazy, keeping tests and text-mode development reproducible.
- SQLite transactions make correction writes atomic.
- The deterministic baseline remains available if learned model loading fails during deployment.
- Users can inspect and delete personalization state.

## Scaling path

The prototype intentionally avoids claiming fleet-scale operation. A production extension would
separate GPU inference from the API, use sticky routing or a shared feature store for profiles,
persist bandit updates through an event log, batch compatible inference requests, and propagate
deadlines so post-editing can fall back rather than exceed the interaction latency budget.
