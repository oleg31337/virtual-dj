# TTS service + engine-agnostic API — design and rollout plan

Status: implemented, verified (unit suites twice, container E2E twice, live deploy on
2026-11-07). Owner: Oleg. Written 2026-11-06.

## 1. Goal

Move speech synthesis out of the Virtual DJ container into its own container that
publishes a **universal HTTP API**. The app must stop knowing anything about
Piper: it asks the service *what it can do* (voices, languages, parameters),
renders the DJ card from that answer, and sends text + the user's chosen values.

Hard requirements from Oleg:

- Default engine **Supertonic 3**, always **5 diffusion steps** (not a UI knob).
- The web UI lets the user pick an **English or Russian voice**.
- The API stays valid if the engine is swapped again later → the app is written
  against the *contract*, never against an engine.
- The service declares its controllable parameters (speed, expressiveness, …);
  the app configures its GUI card from that declaration.
- Everything tested at least twice.

## 2. Architecture

```
┌──────────────────────┐        HTTP/JSON + audio/wav        ┌────────────────────────┐
│ virtual-dj container │ ──────────────────────────────────▶ │  tts container         │
│  app/server.py       │  GET  /v1/capabilities              │  FastAPI (:9000)       │
│  app/tts_client.py   │  GET  /v1/health                    │   ├─ engines/supertonic│
│  app/dj.py           │  POST /v1/speech                    │   └─ engines/piper     │
│  web/ (dynamic card) │  POST /v1/voices/{id}/install        │  models on /models vol │
└──────────────────────┘                                     └────────────────────────┘
        │ ffmpeg loudnorm → mp3 (app-side mastering, unchanged)
        ▼
   Icecast / stream.mp3
```

The app keeps **all** text shaping (LLM prompts, sentence cap, year/number
spelling, Cyrillic transliteration for English voices) and **all** audio
mastering (loudnorm + gain + mp3 encode). The service only turns text into a
float32 WAV. That split is what keeps engine swaps cheap.

## 3. HTTP API v1 — the contract

Base URL from `VDJ_TTS_URL` (compose: `http://tts:9000`).

### `GET /v1/health`

```json
{
  "ok": true,
  "engine": {"id": "supertonic3", "name": "Supertonic 3"},
  "model_ready": true,
  "voices_ready": 10,
  "chars_rendered": 412,
  "renders": 7,
  "last_render_ms": 1830,
  "uptime_s": 512
}
```

`ok` is false when the model/voices are not usable; `detail` then says why. The
app surfaces `detail` verbatim in the UI instead of inventing its own wording.

### `GET /v1/capabilities` — the only thing the GUI needs

```json
{
  "api": {"version": "1", "service": "virtual-dj-tts"},
  "engine": {"id": "supertonic3", "name": "Supertonic 3", "version": "3",
             "license": "openrail", "runtime": "onnxruntime 1.31.1"},
  "defaults": {"voice": "M2", "language": "ru", "params": {"speed": 1.0, "sentence_pause": 0.3}},
  "languages": [{"code": "en", "name": "English"}, {"code": "ru", "name": "Russian"}],
  "voices": [
    {"id": "M2", "name": "Male 2", "gender": "male", "languages": ["en", "ru"],
     "default_language": "ru", "installed": true,
     "description": "Warm male voice; the best Russian of the ten styles.",
     "note": "preferred for Russian"}
  ],
  "params": [
    {"key": "speed", "label": "Speaking speed", "type": "float",
     "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.0, "unit": "×",
     "help": "1.0 = the model's natural pace.", "group": "prosody"},
    {"key": "sentence_pause", "label": "Pause after each sentence", "type": "float",
     "min": 0.0, "max": 2.0, "step": 0.05, "default": 0.3, "unit": "s",
     "help": "Silence inserted between sentence chunks.", "group": "prosody"}
  ],
  "output": {"formats": ["wav"], "sample_rates": [44100], "channels": [1]},
  "text": {"stress_marks": false, "ssml": false, "phonemes": false}
}
```

Parameter `type` is one of `float` (min/max/step), `int` (min/max/step),
`enum` (choices `[{value,label}]`), `bool`. Any other declared field is
informational. **The app renders whatever is declared and never hardcodes a
key**; unknown types are skipped, a missing `params` list means "no controls".

`text.stress_marks` tells the app whether to keep the combining-acute Russian
stress marks it currently emits for espeak (Supertonic: false → the app strips
them; Piper: true → keeps them).

### `POST /v1/speech`

```json
{"text": "Далее — Кино, Звезда по имени Солнце.",
 "voice": "M2", "language": "ru",
 "params": {"speed": 1.0, "sentence_pause": 0.3}}
```

→ `200 audio/wav` (mono, 44.1 kHz, float→int16 PCM), headers:

```
X-TTS-Engine: supertonic3
X-TTS-Voice: M2
X-TTS-Language: ru
X-TTS-Duration: 4.82
X-TTS-Sample-Rate: 44100
X-TTS-Render-Ms: 1183
X-TTS-Params-Used: {"speed":1.0,"sentence_pause":0.3}
X-TTS-Params-Ignored: ["expressiveness"]
X-TTS-Params-Clamped: ["speed"]
```

Errors are JSON `{"detail": {"code": "...", "message": "...", ...}}`:

| status | code | when |
|---|---|---|
| 400 | `bad_request` | empty text, unknown language, text too long |
| 404 | `unknown_voice` | voice id not offered by the engine |
| 409 | `voice_not_installed` | voice exists but its files are missing |
| 422 | `invalid_param` | a declared param has a value of the wrong type |
| 503 | `model_not_ready` | model still downloading / failed to load |

**Lenient by design** (Oleg's preference): an unknown param *key* is ignored and
reported in `X-TTS-Params-Ignored`; an out-of-range *value* is clamped into
`[min,max]` and reported in `X-TTS-Params-Clamped`. Only a wrong *type* is a
422. This means a stale `params` dict from a previous engine never breaks a
break.

### `POST /v1/voices/{voice_id}/install`

Idempotent; downloads whatever the engine needs for that voice. Returns
`{"voice": id, "installed": true, "bytes": 401408000, "already": false}`.
Long-running (up to a few minutes) — the client uses a 10-minute timeout.

## 4. Engine adapter contract

`tts/engines/base.py`:

```python
class Engine(Protocol):
    id: str
    name: str
    version: str
    license: str
    runtime: str

    def languages(self) -> list[dict]          # [{code, name}]
    def voices(self) -> list[dict]             # [{id, name, gender, languages, description, ...}]
    def params(self) -> list[dict]             # the parameter schema above
    def text_support(self) -> dict             # {stress_marks, ssml, phonemes}
    def sample_rate(self) -> int
    def ensure_model(self, progress=None)      # make the model usable (download/load)
    def model_ready(self) -> tuple[bool, str | None]
    def install_voice(self, voice_id: str)     # no-op when the model ships every voice
    def synthesize(self, text, voice, language, params) -> tuple[np.ndarray, int]
```

Engine selection: `TTS_ENGINE=supertonic3|piper` (default `supertonic3`).
A registry maps id → module so adding an engine is one file + one entry.

## 5. Engines

### Supertonic 3 (default)

- Model: HF `Supertone/supertonic-3` (`config.json`, `onnx/*.onnx`, `onnx/*.json`,
  `voice_styles/*.json`), 383 MB, downloaded into `/models/supertonic3`.
- Voices: `M1…M5`, `F1…F5` (10 styles). Every style speaks every language; the
  language is a per-request tag, so all voices declare `["en", "ru"]`.
- Params: `speed` (0.5–2.0, default 1.0 — the helper's own default is 1.05),
  `sentence_pause` 0.0–2.0 s (the helper's `silence_duration`, default 0.3).
- `TTS_STEPS=5` fixed in env/config; **not** exposed as a parameter (a 2-step
  "fast mode" garbles words — measured 54–77 % RU WER).
- Text sanitising in the adapter (measured against `unicode_indexer.json`):
  characters with index `-1` (em dash, ellipsis, non-breaking hyphen …) are
  replaced by the helper's own punctuation map or a space before indexing; `ё`
  and `й` survive via NFKD; combining acute is indexed but is stripped because
  the model has no notion of espeak stress (see `text.stress_marks`).
- Determinism: noise is `np.random.randn`, so renders differ run to run — the
  app's cache key therefore covers text + voice + language + params.

### Piper (second engine, proves the contract)

- Voices from `rhasspy/piper-voices` (reuse the URL scheme already in
  `app/voices.py`, moved into the service).
- Params: `speed`, `expressiveness` (`noise_scale`), `rhythm` (`noise_w`),
  `sentence_pause` (`sentence_silence`) — exactly the knobs today's card shows,
  now described by the service instead of hardcoded in the app.
- Languages: `en`, `ru`; voices are language-specific, so each voice declares one.
- `text.stress_marks: true` (espeak obeys the combining acute).

## 6. App changes

| file | change |
|---|---|
| `app/tts_client.py` (new) | capabilities/health/speech/install client with a 60 s capability cache, retry-once on transient errors, and a `problem()` that explains *why* the DJ is silent (service down, voice missing, model not ready). |
| `app/dj.py` | `synthesize()` → wav from the client, then the existing loudnorm/gain/mp3 chain. Delete `piper_binary()`, `voice_model_path()`, `VOICE_PROFILES`. `voice_language()` → `dj.language` validated against the selected voice. Add `_spell_numbers_ru()`: Russian years/numbers written out as words (Supertonic garbles digits: "1989" → "979"/"1089"). Stress marks are stripped when the engine says `stress_marks: false`. |
| `app/voices.py` | becomes a thin CLI over the service's `/v1/voices/*/install` (no HF logic of its own). |
| `app/config.py` | new keys `dj.language`, `dj.tts_params` (dict); `dj.speed` etc. migrate into `dj.tts_params` on load. `tts.base_url` (env `VDJ_TTS_URL`) stays a config key so the UI can point at another host. |
| `app/server.py` | `GET /api/dj/voices` → voices+languages+params from the service; new `GET /api/dj/tts` (capabilities + current values); `POST /api/dj/voices/install` (was `…/download`); `POST /api/dj/preview` takes `{text, voice, language, params}`; `GET /api/health` includes the service's own health. |
| `web/app.js` + `index.html` | DJ card: engine badge, voice select, language select (only the selected voice's languages), and a **rendered-from-schema** block of parameter controls; "Test this voice" uses the current UI values. Piper-only sliders are deleted. |

Config migration (one-time, on load): `dj.speed → dj.tts_params.speed`,
`dj.noise_scale → dj.tts_params.expressiveness`, `dj.noise_w →
dj.tts_params.rhythm`, `dj.sentence_silence → dj.tts_params.sentence_pause`,
`dj.voice` kept when the engine offers it (else the engine default), and
`dj.language` derived from the old voice id (`ru_RU-*` → `ru`).

## 7. Deployment

- `tts/Dockerfile` — `python:3.13-slim`, `onnxruntime` + `fastapi` + `uvicorn` +
  `huggingface_hub` (+ `piper-tts` for the optional engine), non-root user,
  `/models` volume, healthcheck on `/v1/health`.
- `docker-compose.yml` — new `tts` service (`TTS_ENGINE`, `TTS_STEPS=5`,
  `TTS_AUTOLOAD=1`, `TTS_LANGUAGES=en,ru`, port published **loopback only**,
  `tts-models` volume); app gains `VDJ_TTS_URL=http://tts:9000` and
  `depends_on: tts`.
- App image drops `piper-tts` (the engine now lives in the tts image).

## 8. Tests (each run at least twice)

1. **Service contract tests** (`tts/tests/`): capabilities shape, param schema
   validation, clamping/ignoring, error codes, `install` idempotency — against a
   stub engine (no model needed).
2. **Service live tests**: real Supertonic container — capabilities, EN and RU
   render (duration > 0, ASR round-trip), the same with `TTS_ENGINE=piper`.
3. **App unit tests**: `tts_client` against a fake HTTP server (capabilities,
   caching, error mapping), `_spell_numbers_ru`, config migration, dynamic-card
   schema rendering logic, existing suite green.
4. **End-to-end**: `docker compose up` on fresh volumes → `/api/dj/tts` lists
   voice/language/params → `/api/dj/preview` returns playable audio (checked
   with ffprobe + ASR) → save config → restart → values persist and the DJ card
   shows them → on-air break renders.

Pass = same result on two consecutive runs, second run with a warm cache.

`scripts/e2e_tts.sh` runs checks 1–7 of the container E2E against a live stack
(`scripts/e2e_tts.sh [app_base] [tts_base]`, artifacts in `$E2E_OUT`), so the
whole path can be re-verified after any change without a browser.

## 9. Verification results (2026-11-07)

**Service unit tests** (`tts/tests/`, stub engines, no model): 28 passed, twice.

**Service live smoke** (`tts/scripts/smoke.py`, real model, twice): health `ok`,
`steps=5` in the response headers, 10 voices, English 9.47 s / Russian 8.57 s at
44.1 kHz, unknown parameters ignored and reported in `X-TTS-Params-Ignored`,
out-of-range clamped, install idempotent. ASR round-trip: English ~0 % WER,
Russian intelligible.

**App suite**: 487 → 517 tests, green twice on the final tree.

**Container E2E** (`docker compose -p vdje2e`, own volumes, ports 18420/19000,
twice — cold then warm):

| check | result |
|---|---|
| `GET /v1/health` | `ok:true`, `steps:5`, `model_ready:true`, 10 voices |
| `GET /v1/capabilities` | voices M1–M5/F1–F5 (all installed), languages en+ru, params `speed` 0.5–2.0, `sentence_pause` 0.0–2.0 |
| `POST /v1/speech` (ru) | 4.25 s WAV, `X-TTS-Steps: 5`, `X-TTS-Params-Ignored: ["nonsense"]` |
| `GET /api/dj/tts` | engine `supertonic3`, `current` = configured voice + language, schema-driven params, `problem: null` |
| `POST /api/dj/preview` (ru) | text shaped (`1989 году` → `тысяча девятьсот восемьдесят девятом году`, `120 BPM` → `сто двадцать ударов в минуту`), mp3 44.1 kHz stereo 7.42 s |
| cache | identical request → same file, service render counter unchanged; new text → new render |
| on-air break (`prepare_break` with the LLM stubbed) | mp3 7.50 s into the DJ cache, row written |

**Live deploy on the station's own volumes** (`docker compose up -d --build`,
default project): the `tts` container downloaded the model into `tts-models` and
came up in ~35 s; the app reported the exact reason for the ~20 s window before
that (`TTS service at http://tts:9000: model files are missing…`) and kept
playing music. The legacy Piper config (`voice: en_GB-alan-medium`,
`speed/noise_scale/noise_w/sentence_silence`) migrates in memory to
`tts_params: {speed, expressiveness, rhythm, sentence_pause}` and resolves to the
station's own pick (`M2`, `ru`).
