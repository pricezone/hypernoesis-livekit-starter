# HyperNoesis LiveKit starter agents

Starter voice agents for **Self-Hosted LiveKit on HyperNoesis**: one-click agents
that run on your own LiveKit server, with your own provider keys.

| Folder            | Framework                         | Status      |
| ----------------- | --------------------------------- | ----------- |
| `livekit-agents/` | LiveKit Agents 1.8 (Python)       | available   |
| `pipecat/`        | Pipecat 1.17 (Python)             | available   |

Images (same version tag for both):

- `ghcr.io/pricezone/hypernoesis-livekit-starter:<version>` (LiveKit Agents)
- `ghcr.io/pricezone/hypernoesis-livekit-starter-pipecat:<version>` (Pipecat)

Both starters read the same settings document, choose providers by the same
rules and open with the same AI disclosure; they differ in how they get into a
room (see [the Pipecat starter](#the-pipecat-starter)).

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

## The Pipecat starter

`pipecat/` is the same voice agent built on [Pipecat](https://github.com/pipecat-ai/pipecat)
(BSD 2-Clause) with Pipecat's `LiveKitTransport`: speech-to-text → LLM →
text-to-speech, the Silero VAD, Greek and English, the AI disclosure first. It
uses the [settings contract](#settings-contract-agent-config), the
[provider selection](#provider-selection) and the variables in
[Environment](#environment) exactly like the LiveKit Agents starter, except
dispatch: there is no `HN_AUTO_DISPATCH`, and `HN_AGENT_NAME` only names the
worker (see below).

Differences:

- **Dispatch by webhook.** Pipecat has no LiveKit agent dispatch, so the worker
  dispatches itself from LiveKit webhooks (below).
- **End of turn.** English turns end with Pipecat's local Smart Turn v3 model.
  Its language list does not include Greek, so Greek and `auto` turns end on
  silence: 0.2 s for the VAD plus a 0.5 s pause, once the transcript is in.
- **Speech recognition.** ElevenLabs uses Scribe v2 realtime. OpenAI-compatible
  endpoints (OpenAI, the HyperNoesis API, `HN_STT_URL`) get one request per
  utterance cut by the VAD; with `language: auto` the request carries no
  language, so the endpoint detects Greek or English.
- **Local speech synthesis.** `HN_TTS_URL` must answer `/v1/audio/speech` with
  `response_format: "pcm"` (24 kHz, 16-bit mono), as OpenAI does.
- **Defaults.** "Plugin default" in the provider table means the Pipecat
  service's default here: Gemini `gemini-3.6-flash`, Cartesia `sonic-3.6`,
  Deepgram Aura `aura-2-helena-en`, OpenAI voice `alloy`. Without a `voiceId`,
  ElevenLabs and Cartesia use the same premade voices as the LiveKit Agents
  starter.
- The image has no LiveKit Agents components, so the LiveKit Model License does
  not apply to it.

### Run it

```sh
docker build -f pipecat/Dockerfile -t hypernoesis-livekit-starter-pipecat .
docker run --rm --read-only --tmpfs /tmp -p 8081:8081 \
  -e LIVEKIT_URL=wss://your-livekit-host \
  -e LIVEKIT_API_KEY=... -e LIVEKIT_API_SECRET=... \
  -e OPENAI_API_KEY=... -e ELEVENLABS_API_KEY=... -e DEEPGRAM_API_KEY=... \
  hypernoesis-livekit-starter-pipecat
```

The container runs as uid/gid 10001, writes only to `/tmp` and contains no
secret. Locally: `cd pipecat && uv sync && uv run python -m hn_pipecat`
(`.env.example` lists every variable).

### How it gets into rooms

Add the worker to LiveKit's webhook list, next to any other URL:

```yaml
webhook:
  api_key: <a key from LiveKit's keys>   # the worker must have this key and its secret
  urls:
    - http://<worker-host>:8081/webhook
```

1. Every webhook is verified before it is used: the `Authorization` JWT must be
   signed (HS256) with the secret of a key the worker has, not be expired, and
   carry the SHA-256 of the exact body. Anything else gets `401` and is logged
   with the reason only. Repeated deliveries (same event id) are ignored.
2. `room_started`, or a caller joining (`participant_joined` of a SIP or a
   standard participant), starts a bot in that room when
   - the room name starts with `HN_ROOM_PREFIX` (empty = every room),
   - this worker has no bot in the room yet (one bot per room),
   - no other agent is in the room (checked with LiveKit's room service), and
   - fewer than `HN_MAX_ROOMS` rooms are served (otherwise it is logged and skipped).
3. The bot mints its own token for that room only (identity
   `<HN_AGENT_NAME>-<random>`, participant kind `agent`, one hour), joins, and
   greets the first caller.
4. It leaves when the last caller leaves, when no caller arrives within
   `HN_JOIN_TIMEOUT` seconds, after 5 minutes without speech, on `room_finished`,
   when a provider or the connection fails for good, or when the worker stops
   (`SIGTERM`: every bot leaves, then the process exits).
5. When the worker starts (or restarts) it also joins rooms that already have a
   caller and no agent, so nobody is left waiting.

Use either starter for a given set of rooms, not both: a room that a LiveKit
Agents worker joins after the Pipecat bot would have two agents.

| Variable | Meaning |
| --- | --- |
| `LIVEKIT_URL` | Your LiveKit server (required) |
| `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` (or `HN_AGENT_API_KEY`, `HN_AGENT_API_SECRET`) | Key used to mint room tokens and verify webhooks (required). When both pairs are set, tokens use the first and webhooks signed with either are accepted |
| `HN_AGENT_NAME` | Name in the status and prefix of the bot's identity (default `hypernoesis-pipecat`) |
| `HN_ROOM_PREFIX` | Only rooms whose name starts with this (default: all rooms) |
| `HN_MAX_ROOMS` | Rooms served at the same time (default 8, 1–100) |
| `HN_JOIN_TIMEOUT` | Seconds to wait for a caller after joining (default 60) |
| `HN_HTTP_HOST`, `HN_HTTP_PORT` | Listen address (default `0.0.0.0:8081`) |
| `HN_LOG_LEVEL` | `DEBUG`, `INFO` (default), `WARNING` or `ERROR` |

### Worker status (port 8081)

- `GET /` → `200 OK` once LiveKit answers with the configured key (checked every
  30 s), else `503` with the reason; `503 shutting down` while stopping.
- `GET /worker` → JSON, shaped like the LiveKit Agents worker status:

```json
{
  "agent_name": "hypernoesis-pipecat",
  "worker_type": "JT_ROOM",
  "active_jobs": 1,
  "capacity": 8,
  "load": 0.125,
  "worker_load": 0.125,
  "accepting_jobs": true,
  "sdk_version": "1.17.0",
  "framework": "pipecat",
  "version": "0.1.1"
}
```

`load` = active rooms / `HN_MAX_ROOMS`; `sdk_version` is the Pipecat version.

### Noise cancellation

`noiseCancellation: true` runs the same DeepFilterNet3 build (libDF, pinned
commit, no PyTorch) as a Pipecat input audio filter (`audio_in_filter`), before
the VAD and speech recognition. The pipeline's 16 kHz input is resampled to
48 kHz for the model and back. The model loads in a background thread; audio
passes through unfiltered until it is ready.

## Development

```sh
cd livekit-agents   # or: cd pipecat
uv sync
uv run pytest
uv run ruff check && uv run ruff format --check
```

In both projects the DeepFilterNet test against the real library runs when
`HN_TEST_DFN_LIB` and `HN_TEST_DFN_MODEL` point to a built `libdf.so` and the
model archive.

Images are built on GitHub-hosted runners only (`.github/workflows/image.yml`,
on `v*` tags and manual runs).

## Licence

MIT (see `LICENSE`). The images also contain third-party software under its own
licences: the LiveKit Agents image includes LiveKit's local VAD and turn-detector
models under the LiveKit Model License (usable only with LiveKit Agents); the
Pipecat image includes Pipecat (BSD 2-Clause) with its Silero VAD (MIT) and
Smart Turn (BSD 2-Clause) models. See `THIRD_PARTY_NOTICES`.

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

### Pipecat

Ο φάκελος `pipecat/` είναι ο ίδιος πράκτορας πάνω στο Pipecat (άδεια BSD 2-Clause),
με την ίδια ροή φωνής, τις ίδιες ρυθμίσεις, τους ίδιους κανόνες επιλογής παρόχων
και την ίδια δήλωση «Είμαι ψηφιακός βοηθός.» στην αρχή. Εικόνα:
`ghcr.io/pricezone/hypernoesis-livekit-starter-pipecat`.

- **Πώς μπαίνει στα δωμάτια:** το Pipecat δεν έχει την αποστολή πρακτόρων του
  LiveKit, γι' αυτό ο πράκτορας δέχεται τα webhooks του LiveKit στο
  `http://<διακομιστής>:8081/webhook` (προσθέστε το στα `webhook.urls`). Ελέγχει
  την υπογραφή κάθε webhook με το κλειδί του διακομιστή και μπαίνει σε ένα
  δωμάτιο όταν αυτό ξεκινά ή όταν μπαίνει καλών (SIP ή απλός συμμετέχων): μόνο
  αν το όνομα αρχίζει με το `HN_ROOM_PREFIX`, αν δεν υπάρχει ήδη άλλος πράκτορας
  και αν υπάρχει ελεύθερη θέση (`HN_MAX_ROOMS`). Ένας πράκτορας ανά δωμάτιο.
- **Πότε φεύγει:** όταν φύγει ο τελευταίος καλών, αν δεν έρθει κανείς μέσα σε
  `HN_JOIN_TIMEOUT` δευτερόλεπτα, μετά από 5 λεπτά σιωπής, όταν κλείσει το δωμάτιο
  ή όταν σταματήσει ο πράκτορας.
- **Τέλος ομιλίας:** στα αγγλικά με το τοπικό μοντέλο Smart Turn v3 του Pipecat·
  αυτό δεν υποστηρίζει ελληνικά, οπότε στα ελληνικά η σειρά κλείνει με τη σιωπή
  (περίπου 0,7 δευτερόλεπτα).
- **Κατάσταση:** `GET /` (200 όταν απαντά το LiveKit) και `GET /worker` (JSON με
  `agent_name`, `load`, `active_jobs`, `sdk_version`, `worker_type`) στη θύρα 8081.
- **Ασφάλεια:** τρέχει ως χρήστης 10001, γράφει μόνο στο `/tmp` και δουλεύει με
  σύστημα αρχείων μόνο για ανάγνωση. Δεν περιέχει κλειδιά.
- **Μείωση θορύβου:** ίδια DeepFilterNet3, ως φίλτρο ήχου του Pipecat.

Άδεια: MIT. Το «LiveKit» είναι εμπορικό σήμα της LiveKit, Inc.· το έργο αυτό δεν
συνδέεται με τη LiveKit, Inc. ούτε έχει την έγκρισή της.
