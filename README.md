# HyperNoesis LiveKit starter agents

Starter voice agents for **Self-Hosted LiveKit on HyperNoesis**: one-click agents
that run on your own LiveKit server, with your own provider keys.

| Folder            | Framework                         | Status      |
| ----------------- | --------------------------------- | ----------- |
| `livekit-agents/` | LiveKit Agents 1.8 (Python)       | available   |
| `pipecat/`        | Pipecat (Python)                  | coming soon |

The image is published as `ghcr.io/pricezone/hypernoesis-livekit-starter:<version>`.

> "LiveKit" is a trademark of LiveKit, Inc. This project is not affiliated with or
> endorsed by LiveKit, Inc.

[Ελληνικά](#ελληνικά) below.

---

## What the LiveKit Agents starter does

- Voice pipeline: speech-to-text → LLM → text-to-speech, with the Silero VAD and
  LiveKit's local multilingual turn detector (`turn-detector-v1-mini`). Everything
  runs on your server; nothing is sent to LiveKit Cloud.
- Greek and English. The first words are an AI disclosure ("Είμαι ψηφιακός
  βοηθός." / "I am a digital assistant.") unless you turn it off.
- Settings (prompt, greeting, language, models, voice) come from HyperNoesis at the
  start of every session, so editing them needs no redeploy.
- Provider keys are environment variables on **your** server. They are never
  logged and never sent anywhere except to that provider.
- Optional noise cancellation with DeepFilterNet3 on the CPU.

## Run it

With Docker (from the repository root):

```sh
docker build -f livekit-agents/Dockerfile -t hypernoesis-livekit-starter .
docker run --rm \
  -e LIVEKIT_URL=wss://your-livekit-host \
  -e LIVEKIT_API_KEY=... -e LIVEKIT_API_SECRET=... \
  -e OPENAI_API_KEY=... -e ELEVENLABS_API_KEY=... -e DEEPGRAM_API_KEY=... \
  hypernoesis-livekit-starter
```

Locally with [uv](https://docs.astral.sh/uv/):

```sh
cd livekit-agents
uv sync
cp .env.example .env.local   # fill in LiveKit and at least one key per stage
set -a; . ./.env.local; set +a
uv run python -m hn_agent dev      # or: start
```

The agent registers under the name `hypernoesis-starter` (explicit dispatch: SIP
dispatch rules and room tokens name the agent). Set `HN_AUTO_DISPATCH=true` to
join every new room instead.

### Worker status (port 8081)

LiveKit Agents serves this itself in `start` mode:

- `GET /` → `200 OK` once the worker is registered with LiveKit, `503` before.
- `GET /worker` → JSON: `worker_type`, `agent_name`, `active_jobs`,
  `sdk_version`, `worker_load`, `protocol_version`.

(In `dev` mode the port is random.)

## Environment

| Variable | Meaning |
| --- | --- |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | Your LiveKit server (required) |
| `HN_AGENT_NAME` | Agent name for dispatch (default `hypernoesis-starter`) |
| `HN_AUTO_DISPATCH` | `true` = automatic dispatch to every room (agent name is then empty) |
| `HN_AGENT_CONFIG_URL`, `HN_AGENT_TOKEN` | Where to read the settings (below) |
| `HN_CACHE_DIR` | Directory for the 60-second settings cache file (default: system temp) |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | OpenAI, or any OpenAI-compatible endpoint |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | Google Gemini |
| `DEEPGRAM_API_KEY` | Deepgram STT and Aura TTS |
| `ELEVENLABS_API_KEY` | ElevenLabs Scribe STT and TTS |
| `SPEECHMATICS_API_KEY` | Speechmatics STT |
| `CARTESIA_API_KEY` | Cartesia TTS |
| `HN_STT_URL`, `HN_STT_MODEL` | Free local speech recognition (OpenAI-compatible `/v1/audio/transcriptions`) |
| `HN_TTS_URL`, `HN_TTS_MODEL`, `HN_TTS_VOICE` | Free local speech synthesis (OpenAI-compatible `/v1/audio/speech`) |
| `HN_GATEWAY_URL`, `HN_GATEWAY_TOKEN` | HyperNoesis API (admins only): OpenAI-compatible base URL for LLM, STT and TTS |
| `HN_DFN_LIB`, `HN_DFN_MODEL`, `HN_NC_ATTENUATION_DB` | DeepFilterNet paths (set by the image) and attenuation limit in dB (default 100) |
| `LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE` | `0` skips preloading the turn-detector weights (~450 MB RAM) |

`HN_STT_URL`, `HN_TTS_URL` and `HN_GATEWAY_URL` are base URLs; a trailing
`/audio/transcriptions` or `/audio/speech` is stripped and a bare host gets `/v1`.

## Settings contract (`agent-config`)

At the start of each session the agent calls

```
GET ${HN_AGENT_CONFIG_URL}
Authorization: Bearer ${HN_AGENT_TOKEN}
Accept: application/json
```

and expects `200` with a JSON object. Settings are not secrets; provider keys are
never part of this document.

```json
{
  "prompt": "string",
  "greeting": "string",
  "language": "el | en | auto",
  "llm": { "provider": "auto | openai | anthropic | google | hypernoesis", "model": "string", "baseUrl": "string" },
  "stt": { "provider": "auto | deepgram | elevenlabs | openai | speechmatics | hypernoesis | local", "model": "string" },
  "tts": { "provider": "auto | elevenlabs | cartesia | openai | deepgram | hypernoesis | local", "voiceId": "string", "model": "string" },
  "noiseCancellation": false,
  "aiDisclosure": { "enabled": true, "text": "string" }
}
```

Rules:

- Every field is optional; unknown fields are ignored. A missing or invalid field
  falls back to its default, field by field; invalid fields are logged by name only.
- Defaults: built-in prompt, `language: "auto"` (Greek and English),
  all providers `auto`, `noiseCancellation: false`, `aiDisclosure.enabled: true`.
- `prompt`: up to 20,000 characters; empty or missing = built-in prompt. The agent
  appends a language rule and, with disclosure on, a rule never to claim to be human.
- `greeting`: up to 1,000 characters, spoken verbatim. Missing = built-in greeting;
  `""` = say nothing after the disclosure.
- `aiDisclosure.text`: up to 500 characters; replaces the default line
  ("Είμαι ψηφιακός βοηθός." / "I am a digital assistant." / both for `auto`). The
  disclosure is spoken first, cannot be interrupted, and is not repeated if the
  greeting already contains it. With `enabled: false` the duty to inform callers
  that they talk to an AI (EU AI Act art. 50) moves to you.
- `model`, `voiceId`: ids of up to 200 characters (`[A-Za-z0-9][A-Za-z0-9._:/@+-]*`).
  They apply only when the chosen provider is the requested one (or `auto`).
- `llm.baseUrl`: can only select an endpoint already configured on the server
  (`OPENAI_BASE_URL` or the official OpenAI URL). A key is never sent to a URL that
  is not set next to it in the server's environment; a different value is ignored
  with a warning.
- Cache: 60 seconds, shared by all sessions in the container (memory + file).
  If the server cannot be reached (timeout 3 s, non-200, redirect, invalid JSON,
  over 256 KiB) the agent uses the last good copy, or the built-in defaults when
  there is none, logs the reason once, and does not ask again for 60 seconds.
- `HN_AGENT_CONFIG_URL` must be `https`, except for loopback, private addresses and
  single-label container names. If the URL or token is unset, the defaults are used
  (logged once).

## Provider selection

A provider is **available** when its key (or service URL) is set on the server.
`auto` takes the first available one in this order — own keys, then the
HyperNoesis API, then the free local services:

| Stage | `auto` order | Default model |
| --- | --- | --- |
| LLM | `openai` → `anthropic` → `google` → `hypernoesis` | `gpt-4.1-mini`, `claude-haiku-4-5`, plugin default (Gemini), `gpt-4.1-mini` |
| STT | `deepgram` → `elevenlabs` → `openai` → `speechmatics` → `hypernoesis` → `local` | `nova-3`, plugin default (Scribe realtime), `gpt-4o-mini-transcribe`, plugin default, `gpt-4o-mini-transcribe`, `whisper-1` (or `HN_STT_MODEL`) |
| TTS | `elevenlabs` → `cartesia` → `openai` → `deepgram` → `hypernoesis` → `local` | `eleven_flash_v2_5`, plugin default (Sonic), `gpt-4o-mini-tts`, plugin default (Aura), `tts-1`, `tts-1` (or `HN_TTS_MODEL`) |

- A provider named in the settings but not available falls back to the `auto`
  order (logged), with that provider's default model.
- With `OPENAI_BASE_URL` set to a non-OpenAI endpoint, `auto` does not use it for
  speech (such endpoints rarely have audio APIs); naming `openai` for STT/TTS
  explicitly does.
- There is no free local LLM: without an LLM key (or the HyperNoesis API) the
  session ends with a log line saying which variables to set. The same happens
  for STT/TTS when neither a key nor `HN_STT_URL`/`HN_TTS_URL` is set.
- File-based STT (OpenAI-compatible endpoints) is wrapped in LiveKit's
  `StreamAdapter`, cut into utterances by the VAD.
- `HN_GATEWAY_URL` must be `https` (except loopback/private hosts).
- Voice: `tts.voiceId` (ElevenLabs voice id, Cartesia voice id, OpenAI voice name,
  Deepgram Aura model such as `aura-2-thalia-en`); empty = the plugin's default voice.

### Language

| `language` | STT | Turn detection |
| --- | --- | --- |
| `el` | Greek | VAD silence only |
| `en` | English | LiveKit turn detector |
| `auto` | detected by ElevenLabs, OpenAI and the local service; **Greek** for Deepgram and Speechmatics (no Greek+English detection) | turn detector for its languages, VAD silence for Greek turns |

LiveKit's turn detector (`turn-detector-v1-mini`, and the older multilingual text
model) does **not** support Greek. Its language list is ar, de, en, es, fr, hi, id,
it, ja, ko, nl, pt, tr, zh. Greek turns therefore end on VAD silence (about half a
second after the caller stops speaking).

## Noise cancellation (DeepFilterNet3)

`noiseCancellation: true` runs DeepFilterNet3 (MIT OR Apache-2.0, code and
weights) on the caller's audio before VAD, turn detection and STT. The image builds
DeepFilterNet's C library (`libDF`, `capi` feature) from a pinned commit: inference
runs in pure Rust (`tract`) on the CPU, so there is no PyTorch and no onnxruntime.
Audio is taken at 48 kHz mono and filtered in 10 ms hops.

Measured on a busy 4-core workstation: about 18 ms of CPU per 50 ms of audio
(≈ 0.35 of one core per call), 1.3–2.8 s to load the model (in a background
thread; audio passes through unfiltered until it is ready), about 95 MB RAM.
It is off by default and not yet compared against Krisp.

## Development

```sh
cd livekit-agents
uv sync
uv run pytest
uv run ruff check && uv run ruff format --check
```

The DeepFilterNet test against the real library runs when `HN_TEST_DFN_LIB` and
`HN_TEST_DFN_MODEL` point to a built `libdf.so` and the model archive.

Images are built on GitHub-hosted runners only (`.github/workflows/image.yml`,
on `v*` tags and manual runs).

## Licence

MIT (see `LICENSE`). The image also contains third-party software under its own
licences, including LiveKit's local VAD and turn-detector models under the LiveKit
Model License (usable only with LiveKit Agents). See `THIRD_PARTY_NOTICES`.

---

## Ελληνικά

Έτοιμοι φωνητικοί πράκτορες για το **Self-Hosted LiveKit της HyperNoesis**. Τρέχουν
στον δικό σας διακομιστή LiveKit, με τα δικά σας κλειδιά παρόχων.

- **Ροή φωνής:** αναγνώριση ομιλίας → γλωσσικό μοντέλο → σύνθεση ομιλίας, με
  Silero VAD και τον τοπικό ανιχνευτή τέλους ομιλίας του LiveKit. Όλα τρέχουν στον
  διακομιστή σας, τίποτα δεν πάει στο LiveKit Cloud.
- **Ελληνικά και Αγγλικά.** Η πρώτη φράση είναι η δήλωση «Είμαι ψηφιακός βοηθός.»
  (Κανονισμός ΤΝ της ΕΕ, άρθρο 50). Αν την απενεργοποιήσετε, την ευθύνη ενημέρωσης
  των καλούντων την έχετε εσείς.
- **Ρυθμίσεις** (οδηγίες, χαιρετισμός, γλώσσα, μοντέλα, φωνή): έρχονται από τη
  HyperNoesis στην αρχή κάθε συνεδρίας, άρα οι αλλαγές δεν θέλουν νέα εγκατάσταση.
  Δεν περιέχουν κλειδιά.
- **Κλειδιά:** μένουν ως μεταβλητές περιβάλλοντος στον δικό σας διακομιστή. Δεν
  καταγράφονται ποτέ και στέλνονται μόνο στον αντίστοιχο πάροχο.
- **Δωρεάν τοπικές υπηρεσίες:** χωρίς κλειδί αναγνώρισης ή σύνθεσης ομιλίας, ο
  πράκτορας χρησιμοποιεί τις δωρεάν υπηρεσίες `HN_STT_URL` / `HN_TTS_URL` του
  διακομιστή σας. Για το γλωσσικό μοντέλο χρειάζεται πάντα κλειδί.
- **Ελληνικά και ανίχνευση τέλους ομιλίας:** ο ανιχνευτής του LiveKit δεν υποστηρίζει
  ελληνικά, οπότε στα ελληνικά η σειρά ομιλίας κλείνει με τη σιωπή (VAD).
- **Μείωση θορύβου:** προαιρετικά με DeepFilterNet3 στον επεξεργαστή, χωρίς
  PyTorch. Κλειστή από προεπιλογή· κοστίζει περίπου το ένα τρίτο ενός πυρήνα ανά
  κλήση.
- **Κατάσταση:** `GET /` και `GET /worker` στη θύρα 8081.

Άδεια: MIT. Το «LiveKit» είναι εμπορικό σήμα της LiveKit, Inc.· το έργο αυτό δεν
συνδέεται με τη LiveKit, Inc. ούτε έχει την έγκρισή της.
