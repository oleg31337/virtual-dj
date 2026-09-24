# Virtual DJ

An always-on internet radio station for your own MP3 collection, with an AI DJ
that actually talks about the music.

It scans your library, builds a continuous MP3 stream, and every few tracks a
synthesized DJ voice introduces the next song with real facts about it — pulled
from your file tags and enriched from MusicBrainz/Wikipedia, then written by a
local LLM and spoken by a local neural voice. Nothing leaves your network except
optional metadata lookups.

Point VLC, Winamp, Sonos, or any browser at the stream URL and it just plays.

```
┌──────────┐   scan    ┌──────────┐  facts   ┌─────────┐  script  ┌────────┐
│ /mnt/mp3 │ ────────► │  SQLite  │ ───────► │ Ollama  │ ───────► │ Piper  │
└──────────┘  mutagen  │ library  │ MusicBr. │  (LLM)  │          │ (TTS)  │
                       └────┬─────┘          └─────────┘          └───┬────┘
                            │ next track                   DJ voice   │
                            ▼                                         ▼
                       ┌──────────────────────────────────────────────────┐
                       │  Broadcaster — ffmpeg → frame-aligned MP3 stream │
                       └───────────────────────┬──────────────────────────┘
                                               │  /stream.mp3
                     ┌─────────────────────────┼─────────────────────────┐
                     ▼                         ▼                         ▼
                   VLC                      Sonos                    Browser
```

## Features

- **Real radio stream** — one continuous `/stream.mp3` any player can open. New
  listeners join mid-song and start hearing audio instantly.
- **Talking DJ** — an LLM writes a short intro from the track's actual metadata;
  Piper speaks it. Frequency, length, persona, voice, and speed are all tunable.
- **Grounded facts** — tags first, then MusicBrainz/Wikipedia. The prompt is
  fact-constrained to keep the DJ from making things up.
- **Self-repairing metadata** — when a file has no usable tags, the scanner
  guesses artist/title from the filename and folder, then **confirms the guess
  on the web** (MusicBrainz + iTunes, fully cached). Anything still unnamed is
  registered by its file name with `Unknown` as the artist — **no file is ever
  kept out of playlists for having bad tags**.
- **Broad format support** — MP3, FLAC, OGG/Opus, M4A/MP4, WAV and **WMA**
  (ASF tags included) are all scanned, tagged and streamed.
- **Non-Latin? Kept.** Cyrillic, Greek, Japanese and other scripts are valid
  track names — the scanner keeps them, recovers a romanised form when the
  tags are corrupt (using the local LLM), and lets the DJ announce them.
- **Every track gets a genre.** Web lookups provide a genre when they can; for
  the rest, the local LLM is the guaranteed second opinion. No playable track
  is left with an unknown genre, so themed playlists work for the whole library.
- **Themed programs.** The queue is grouped into DJ-style "programs" — runs of
  tracks sharing a genre, an artist, or a decade — with the DJ announcing each
  vibe switch before the next block. The *Programs* card lists the themes
  (biggest by track count first) and every one of them can be switched off with
  a click: switched-off themes never reach the queue, in any strategy. Program
  size, grouping strategy and how many themes take part are adjustable in
  `playback.program` (config or web UI).
- **Even volume across the whole library.** Every track is measured once
  (EBU R128 loudness + true peak) and then streamed at its own fixed gain, so
  loud songs are pulled down and quiet ones lifted — without the level being
  ridden inside a song. The measurement pass runs in the background at low
  priority, is resumable, and can be watched and tuned in the *Volume
  normalization* card (see [Volume normalization](#volume-normalization)).
- **Web control panel** — live now-playing, library browser/search, genre
  filters, queue editing, presets, and DJ settings. Includes a browser player.
- **Your library, untouched** — the scanner only ever reads your music files.

## Requirements

- Linux, Python 3.11+
- `ffmpeg` and `ffprobe` on PATH (`sudo apt install ffmpeg`)
- [Ollama](https://ollama.com) reachable on your network, for DJ scripts
  (optional — the station plays music fine without it; the DJ just stays
  silent until an Ollama endpoint is configured)
- internet access on first run, to auto-download the Piper voice models
  (~190 MB for the two default voices)

## Install

```bash
git clone <your-repo-url> virtual-dj
cd virtual-dj
./run.sh                      # creates .venv, installs deps, downloads the
                              # default voices, starts on :8420
```

`run.sh` is the canonical launcher. It:

1. creates a Python venv (`.venv`) if missing and installs `requirements.txt`;
2. downloads the default Piper voice the app needs (`dj.voice`) into
   `data/voices/` — so the DJ can speak on a clean clone;
3. launches the server on `0.0.0.0:8420` (override with `--host`/`--port`).

To run the server directly instead: `python -m app.main --host 0.0.0.0 --port 8420`.

> No network at install time? The app still starts — voice download is
> best-effort and retries at runtime. You can fetch voices any time from the
> web UI (DJ Settings → Voice picker → **Test**), or with
> `python -m app.voices [--all]` (or `python -m app.voices en_US-amy-medium`).

Then open **http://localhost:8420**, set your music folder in the Library
panel (or pre-seed it, see below), and hit **Rescan**.

### Pre-seeding the config

Copy the example and edit it before first run:

```bash
cp config.example.json data/config.json   # then edit music_dir / llm / etc.
```

All keys are optional — anything omitted falls back to the defaults in
`app/config.py`. The web UI writes this same `data/config.json` file, so you
rarely need to edit it by hand.

### The DJ voice

The Piper voice models are large binary files (git-ignored) stored in
`data/voices/`. On a **fresh install they are downloaded automatically** — the
configured default (`dj.voice`, `en_US-amy-medium`) is fetched on first run,
and the web UI can download any other curated voice on demand.

Any Piper voice works — drop the `.onnx` + `.onnx.json` pair in `data/voices/`
and pick it in the DJ Settings panel. To grab them manually:

```bash
mkdir -p data/voices && cd data/voices
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
```

The full catalogue (Amy, Lessac, LibriTTS-R, Ryan, Bryce, plus the Russian
Irina/Denis/Dmitri/Ruslan) is listed in `app/dj.py` (`VOICE_PROFILES`); every
one is downloadable from the same HuggingFace repo via the web UI or
`python -m app.voices --all`.

**The chosen voice sets the DJ's language.** English voices (`en_US-*`,
`en_GB-*`) make the DJ speak English; picking a Russian voice (Irina, Denis,
Dmitri, Ruslan) makes the DJ speak Russian — intros are written in Russian and
spoken natively, with no transliteration. There is no separate "Russian voice"
setting and no per-song language detection: the whole station speaks the
language of the voice you select in DJ Settings.

## Listening

External players (Winamp / VLC / Sonos) consume the **Icecast mount**, which the
app relays into a managed Icecast2 server running in the same process/container:

| Player  | How |
|---------|-----|
| Winamp  | File → Play URL → `http://<host>:8008/virtualdj` |
| VLC     | Media → Open Network Stream → `http://<host>:8008/virtualdj` |
| Sonos   | Add a radio station by URL in the S2 app → same URL |
| Browser | Just open `http://<host>:8420` and hit **Listen** (uses `/stream.mp3`) |
| CLI     | `mpv http://<host>:8420/stream.mp3` |

Port `8008` is the default Icecast port (change `ICECAST_PORT` in `.env`), and
`virtualdj` is the default mount (change `ICECAST_MOUNT`). The browser player
uses the app's own `/stream.mp3`; only external players need the Icecast mount.

## If the music library is not mounted

The app watches playback, not just the database: a queued file that cannot be
opened is flagged (`missing = 1`, so it is never queued again) and the station
stops rather than spinning.

If the music share is unmounted (or moved), you get a clear warning in the
**Now playing** card — *"N queued files in a row could not be played … is the
music library mounted?"* — and the app backs off progressively (up to 30 s
between attempts) instead of burning through the queue. Remount the share, then
run a **library scan** to restore the affected tracks (the scan clears the flag
for files that are back and removes the rows for files that are gone).

Without those guards an unmounted share is pathological: every miss returns
instantly, so the queue is consumed at ~700 items/second and re-filled forever —
and because each refill stamps new DJ talks, the DJ LLM gets called continuously
(measured: ~34,000 queue items and 13 LLM calls a minute, indefinitely).

## Running with Docker

The app ships a `Dockerfile` and `docker-compose.yml`. Everything the app
writes — `config.json`, the SQLite library DB, the downloaded **Piper voice
models**, and the DJ cache — lives in a **named volume** mounted at `/data`, so
it survives container restarts and upgrades. Your music library is mounted
**read-only** from the host (the DJ never writes to it).

```bash
cp .env.example .env        # edit at least MUSIC_DIR to point at your music
docker compose up -d --build
```

Then open **http://localhost:8420**.

On first start the container **auto-downloads the default voice model**
(English, ~110 MB) into the volume; the DJ can speak out of the box.
It also **auto-scans `/music` on first boot** (the container's `music_dir`
defaults to `/music`, which is where your host library is mounted), so the
station indexes your music with no setup. If `/music` is empty or unmounted,
the Library panel shows a clear "No audio files found" / "Music folder not
found" message instead of failing silently. To rescan after adding files, use
the **Rescan** button in the Library panel (or change the path there).

### What the container does for you

- **Runs as root on purpose**: Icecast2 must start as root so its `<changeowner>`
  can drop to `nobody` — MP3 source mounts only serve when that privilege drop
  happens. The single image owns Icecast end-to-end (renders `icecast.xml` from
  `data/config.json`, supervises the daemon, relays the stream into it), so the
  read-only-rootfs / cap-drop hardening of the old split stack does not apply.
- **Healthcheck** against `/api/health`; the container reports `healthy` once
  the app is serving, and `restart: unless-stopped` keeps it up.
- Application code is read-only; all mutable state lives under `/data` (the
  named volume) and `/tmp` (icecast logs / runtime files).
- Music is mounted read-only at `/music` (override with the `music_dir` config
  in the web UI if you want a different in-container path).

### Tuning (`.env`)

| Key | Default | Meaning |
|-----|---------|---------|
| `HTTP_PORT` | `8420` | Host port published for the UI + stream |
| `MUSIC_DIR` | `/mnt/mp3` | **Host** path to your music (mounted `:ro` at `/music`) |
| `VDJ_LOG_LEVEL` | `info` | Log verbosity |
| `VDJ_NO_VOICE_DOWNLOAD` | `0` | `1` = skip the first-run voice download |
| `VDJ_ICECAST_ENABLED` | `1` | `0` disables Icecast delivery (browser player still works) |
| `ICECAST_PORT` | `8008` | Icecast listen **and** published host port (both sides of the mapping — change together) |
| `ICECAST_MOUNT` | `virtualdj` | Mountpoint external players open (`http://<host>:<port>/<mount>`) |
| `ICECAST_SOURCE_PASSWORD` | `hackme` | Relay password the app's pusher uses to connect to Icecast |
| `ICECAST_PUBLIC_HOST` | _blank_ | Host shown in the web UI's stream URL (blank = derive from the browser) |
| `VDJ_OLLAMA_URL` | _unset_ | Ollama endpoint (e.g. `http://host-gateway:11434` for host Ollama) |
| `VDJ_OLLAMA_MODEL` | _unset_ | Model for DJ scripts |

### Reaching an Ollama on the host

If Ollama runs on the Docker host, point the container at it with
`VDJ_OLLAMA_URL=http://host-gateway:11434` in `.env` (or your host's LAN IP).

### Upgrading

```bash
docker compose pull   # or: docker compose up -d --build
```

Your `data` volume is untouched — config, library DB, and voice models carry
over. The first boot after an upgrade will reuse the already-downloaded voices.

## Configuration

Settings live in `data/config.json` (created on first run, never committed).
The web UI writes this same file — you rarely need to edit it by hand.

A few env vars override the defaults at first boot, useful for containers:

| Variable | Default | Meaning |
|----------|---------|---------|
| `VDJ_MUSIC_DIR` | `~/Music` | Initial music folder |
| `VDJ_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `VDJ_OLLAMA_MODEL` | `qwen3.5:9b` | Model used for DJ scripts |
| `VDJ_DATA_DIR` | `./data` | Where state is kept (config, DB, voices) |
| `VDJ_PORT` | `8420` | Listen port |
| `VDJ_HOST` | `0.0.0.0` | Bind address |
| `VDJ_LOG_LEVEL` | `info` | Log verbosity |
| `VDJ_NO_VOICE_DOWNLOAD` | `0` | Set to `1` to skip the first-run voice download |
| `VDJ_LOUDNESS_WORKERS` | `2` | Parallel loudness analyses (raise near your core count for a faster first pass) |

## Volume normalization

A real library spans a wide loudness range — 17 dB between the quietest and the
loudest file in the library this was built against — so a plain shuffle makes
loud tracks jump out and quiet ones disappear. Virtual DJ measures every track
once and then streams it at its own fixed gain:

```
gain = target_lufs − integrated_lufs                  # −16 − (−20.1) = +4.1 dB
gain = min(gain, true_peak_ceiling − true_peak_dbfs)  # never clip
gain = clamp(gain, min_gain_db, max_boost_db)         # never shout
```

Measurement is EBU R128 (`ffmpeg -af ebur128=peak=true`) over the **first 120
seconds** of each file — the cheapest representative sample — and the result
(integrated LUFS, true peak, gain) is cached per track, so each file is measured
once and re-measured only when the file or the settings change. Measured on the
reference host, one file takes ~390 ms at the default 2 workers and ~170 ms at
6; for an 8,800-file library that is a **one-off pass of ~1 hour at 2 workers**
(or ~25–30 minutes at 4–6 — scaling flattens past 4), and effectively nothing
afterwards for new files. Set `VDJ_LOUDNESS_WORKERS` in `.env` (or the card's
*Workers* field) to change it.

The pass is **background and gradual**: a small nice'd worker pool drains the
"not measured yet" queue while the station keeps streaming, and each track
starts using its gain the moment it has been measured. Tracks not measured yet
(and everything with the feature switched off) keep the legacy dynamic
`loudnorm` chain, so volume never gets *worse* during the pass. Because the
queue is a database query, restarting the app resumes the pass instead of
starting over.

The web UI's **Volume normalization** card shows library-wide progress
(`6,772 of 10,158 measured (67%)` with a live bar, the ETA and the last analyzed
file) and puts each track's measured gain next to it in the library list. Its
buttons do different things:

| button | what it does |
|---|---|
| **Measure missing (N)** | analyzes only the tracks that have no measurement yet (new files, or a pass that never finished). Shows the pending count; disabled when there is nothing to do. |
| **Stop** | pauses the pass after the file being analyzed; it resumes where it left off. |
| **Re-analyze all** | *discards* every existing measurement and measures the whole library again (only needed after changing the analyzed window, or to refresh stale results). Asks for confirmation first. |

The pass also keeps running across restarts: progress lives in the database, not
in memory, so a container restart resumes the backlog instead of losing it.

| Setting (`loudness.*` in `data/config.json`) | Default | Meaning |
|---|---|---|
| `enabled` | `true` | `false` = everything keeps the dynamic `loudnorm` chain |
| `target_lufs` | `-16` | Perceived-loudness target for music |
| `true_peak_ceiling` | `-1.5` | Post-gain true-peak ceiling (dBTP) |
| `max_boost_db` | `6` | Never amplify a quiet track more than this |
| `min_gain_db` | `-12` | Never attenuate a track more than this |
| `window_seconds` | `120` | Seconds analyzed from the START of a file (0 = whole file) |
| `workers` | `2` | Parallel analyses (each one is single-threaded); set `VDJ_LOUDNESS_WORKERS` in `.env` |
| `autostart` | `true` | Drain the queue at boot and after every scan |

Notes:

- The analyzed window is part of the measurement id stored per track, so
  changing `window_seconds` re-queues the whole library rather than mixing two
  different estimates. "Re-analyze everything" in the card does the same on
  demand.
- A shorter window is faster but noisier: on the reference library a 60 s
  mid-track window estimates a track's loudness within ~0.8 dB on average,
  while the first 120 s averages ~1.4 dB (up to ~4 dB on tracks that start with
  a long quiet intro). Set `window_seconds: 0` to measure whole files exactly.
- Files that cannot be decoded (or are silent) are marked measured with no gain:
  they keep the dynamic fallback and are never retried in a loop.
- A row whose file is gone from disk (a stale index) is stamped the same way and
  counted separately in the card — the next rescan removes such rows entirely.
- The DJ voice is deliberately excluded: every spoken clip comes from the same
  Piper model, so its level is already consistent, and `dj.gain_db` remains the
  artistic trim. Music now meets it at a known target instead.

## Running as a service

```bash
sudo cp deploy/virtual-dj.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now virtual-dj
```

Edit `User=`, `WorkingDirectory=`, and the `ExecStart` path first.

## Architecture

| Module | Role |
|--------|------|
| `app/library.py` | Filesystem scan + tag extraction (mutagen) into SQLite |
| `app/enrich.py` | MusicBrainz / Wikipedia lookups, cached on disk |
| `app/dj.py` | Prompt building, Ollama call, Piper synthesis |
| `app/scheduler.py` | Queue, genre/artist filters, shuffle, DJ cadence, prefetch |
| `app/stream.py` | `Broadcaster` — ffmpeg encode, frame alignment, client fan-out |
| `app/server.py` | FastAPI: `/stream.mp3`, REST API, WebSocket now-playing |
| `web/` | Control panel (plain JS, no build step) |

### How the stream stays smooth

Three details do the heavy lifting, and all three are regression-tested:

- **Frame alignment.** Only complete MPEG frames are ever written to a client.
  Emitting a partial frame at a track boundary is what makes VLC report
  `Header missing` and stutter.
- **Burst on connect.** A new listener immediately receives a few seconds of
  buffered audio, so their client's network cache is full before playback
  starts instead of starving.
- **Pacing cushion.** The encoder stays a few seconds ahead of real time, so a
  scheduler hiccup or an ffmpeg spawn never starves a connected player.

The stream endpoint is a raw ASGI response on purpose — Starlette's
`StreamingResponse` disconnects idle-but-connected radio listeners.

## Development

```bash
.venv/bin/python -m pytest tests/ -q
```

The suite covers tag parsing, DJ prompt/fact grounding, scheduler cadence,
voice downloading, and end-to-end streaming against a real server — including
decoding the live stream with ffmpeg across forced track switches to assert
zero decode errors. Tests use temp directories and never touch your library.

> Note for `hermes verify` / `uv` users: activate the venv first
> (`source .venv/bin/activate`), since the detected recipe invokes bare
> `pytest` and `uvicorn`.

## License

MIT
