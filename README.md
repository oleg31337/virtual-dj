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
  (biggest by track count first) and every one can be switched off with a click:
  switched-off themes never reach the queue, in any strategy.
- **Playlist rules that hold.** No band plays more than 2 songs in a row, no
  artist comes back within 10 tracks, no song repeats within 50 plays, and every
  program's length scales with how much its theme actually holds — so a small
  genre gets a short run instead of being skipped. All of them are enforced
  across programs, refills and the flat shuffle, shown in the *Programs* card
  (including when the current selection makes one unenforceable), and tunable
  one by one. See [Playlist rules](#playlist-rules).
- **Genre and decade selection.** The *Genres/decades* card's chips are a
  selection: with nothing picked everything plays, and once you pick genres
  and/or decades only those are queued — in programs, in the flat shuffle and in
  the library browser alike. A selection on the same dimension as the program
  theme also narrows the *rotation*; a cross-dimension selection is ANDed into
  every program's tracks.
- **Artist filter.** The *Genres/decades* card's artist field narrows the queue
  to the artists you list (comma-separated, suggestions from your library, blank
  = every artist). It composes with the genre/decade chips and the search box.
- **Even volume across the whole library.** Every track is measured once
  (EBU R128 loudness + true peak) and then streamed at its own fixed gain, so
  loud songs are pulled down and quiet ones lifted — without the level being
  ridden inside a song. The measurement pass runs in the background at low
  priority, is resumable, and can be watched and tuned in the *Volume
  normalization* card (see [Volume normalization](#volume-normalization)).
- **Web control panel** — live now-playing with the DJ's script, library
  browser/search, the *Genres/decades* selection, queue editing and reordering,
  DJ voice settings with a "test this voice" preview,
  the *Programs* card (themes, rules, program length), presets, the loudness
  card with per-track gains, DJ/voice settings, Icecast control and a running
  report of temporary data with a **Clean now** button. Includes a browser
  player.
- **Your library, untouched** — the scanner only ever reads your music files.

## Requirements

- Linux, Python 3.11+
- `ffmpeg` and `ffprobe` on PATH (`sudo apt install ffmpeg`)
- [Ollama](https://ollama.com) reachable on your network, for DJ scripts
  (optional — the station plays music fine without it; the DJ just stays
  silent until an Ollama endpoint is configured)
- internet access on first run (optional), to auto-download the configured Piper
  voice — the default is `en_GB-alan-medium`, **~60 MB**

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
> `python -m app.voices [--all]` (or `python -m app.voices en_GB-alan-medium`).

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
configured default (`dj.voice`, `en_GB-alan-medium`) is fetched on first run,
and the web UI can download any other curated voice on demand.

Any Piper voice works — drop the `.onnx` + `.onnx.json` pair in `data/voices/`
and pick it in the DJ Settings panel. To grab them manually:

```bash
mkdir -p data/voices && cd data/voices
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json
```

The catalogue is **16 curated voices** — 12 English (Amy, Lessac, LibriTTS-R,
Ryan, Bryce, Sam, Norman, Alan, Jenny, Southern English, HFC male/female) and 4
Russian (Irina, Denis, Dmitri, Ruslan) — listed in `app/dj.py`
(`VOICE_PROFILES`) and downloadable from the same HuggingFace repo via the web
UI. Each is ~63 MB; `python -m app.voices --all` fetches the lot (**~1 GB**).

### How the DJ's text is prepared

Piper reads what it is given, so the *text* decides whether the delivery sounds
like a host or like a catalogue entry. Three things are now enforced:

- **Spelled-out dates and numbers.** The prompt asks the LLM for
  "nineteen ninety-nine", never "1999", and the app spells any bare year that
  still slips through (`1984` -> `nineteen eighty-four`) before synthesis.
  Russian is deliberately different: the Russian voice reads numerals correctly
  on its own, so years stay as digits there.
- **Names in their spoken form.** Stylised spellings are written the way they are
  said — P!nk is "Pink", Ke$ha is "Kesha", Panic! at the Disco is "Panic at the
  Disco", deadmau5 is "dead mouse" — and symbols that are not pronounced are
  dropped. The LLM is told to say them plainly but never to invent a different
  name.
- **Punctuation for breath.** Short clauses; a comma is a small pause, an em dash
  a longer one. Measured: the engine's own punctuation pauses only reach
  0.10–0.26 s, which is why `dj.sentence_silence` (below) adds a real sentence
  gap on top.

#### Russian stress marks: only one character works

The Russian voice guesses stress, and guesses wrong on homographs
(*за́мок* the castle vs *замо́к* the lock). espeak-ng obeys exactly one mark:

| Written | Result |
|---|---|
| `´` **directly after** the stressed vowel (combining acute, U+0301) | correct stress ✅ |
| `´` as a plain acute (U+00B4) — *what the prompt used to ask for* | **silently ignored** ❌ |
| `ʼ` / `ʹ` (modifier letter/prime) | **read aloud as the character's name** 💥 |
| mark written *before* the vowel | ignored ❌ |

So the LLM is asked for a plain `´` after the vowel (easy to produce) and the app
rewrites whatever it gets into the character espeak obeys, moving it to the far
side of the vowel and dropping marks with no vowel to attach to. Ё is always
written as `ё`, never `е`.

**The chosen voice sets the DJ's language.** English voices (`en_US-*`,
`en_GB-*`) make the DJ speak English; picking a Russian voice (Irina, Denis,
Dmitri, Ruslan) makes the DJ speak Russian — intros are written in Russian and
spoken natively, with no transliteration. There is no separate "Russian voice"
setting and no per-song language detection: the whole station speaks the
language of the voice you select in DJ Settings.

## Playlist rules

Five knobs shape the queue, and they are enforced in every mode — themed
programs, the flat shuffle, across refills and across restarts:

| Knob (`playback.*`) | Default | Rule |
|---|---|---|
| `program.max_consecutive_artist` | `2` | No more than 2 songs in a row by one band (`1` = never back-to-back) |
| `artist_gap` | `10` | The same artist may not come back within 10 tracks (`0` = off) |
| `repeat_window` | `50` | A song may not repeat within 50 plays (`0` = off) |
| `program.size` | `6` | The longest program — taken by the biggest theme |
| `program.min_size` | `2` | The shortest program — for the smallest theme |

Three more say *what* the queue is built from: `genres`, `decades` and `artists`
(the *Genres/decades* card), `program.limit` (how many themes take part) and the
per-theme on/off switches in the *Programs* card. All of them live in
`data/config.json` and are editable from the web UI.

### How they interact

- **Programs are contiguous runs of one theme**, and the DJ announces every
  switch — including the first program of a refill, which used to slip through
  unannounced.
- **The artist gap beats the no-long-runs cap.** With a 10-track gap, two songs
  by one band in a row cannot happen at all: the stricter rule wins.
- **A theme that cannot fill its program is skipped** for that round and another
  program is used — "not enough songs for this genre/band, move on". The builder
  tries strict spacing first, then relaxed spacing, then themes that are
  themselves inside the gap, so a small rotation never goes silent.
- **Same dimension narrows, cross dimension ANDs.** Picking the 90s while
  theming programs by decade restricts the rotation to 1990s programs; picking
  the 90s while theming by genre filters each program's tracks instead.
- **Spacing needs artists to space.** The gap needs at least as many distinct
  artists in the selection as the gap itself; with fewer (a single-band filter, a
  one-artist genre) the builder plays the longest-absent artist and the card
  reports **artist spacing relaxed**, plus how often it gave way.
- **The card never claims a rule the queue will break.** `cap_enforceable` and
  `gap_enforceable` come from the same functions the builder uses, so switching
  to a single artist makes the summary say "artist limit relaxed" instead of
  stating a rule that cannot hold.
- **The station never goes silent.** If no theme can fill a program the queue
  falls back to a *filtered* flat shuffle (your selection still applies). Only a
  filter that matches nothing at all falls back to the whole library.

### Measured on the reference library (10,158 tracks)

| What | Result |
|---|---|
| Artist gap, live queue | 63 songs, **0 violations**, closest an artist came back was exactly 10, longest run 1 |
| Artist gap, 10-configuration matrix | held in genre, decade and flat modes; held across program boundaries in artist mode; 18–19 of 20 themes aired per 120 songs, every program exactly its theme's length |
| No long runs | longest run dropped from 6 songs to 2; 19/20 genre themes, 20/20 artist themes, 8/8 decade themes still fill a program |
| Program scaling | Rock (2,840 tracks) runs 6, Electronic (1,093) runs 3, the rest of the rotation runs 2 — live queue `Rock:6 … Jazz:2 · Trip Hop:2` |
| Repeat window | over 600 songs the closest a song came back was 64 plays (211 with a 200-song window) |
| Decade selection | chips 2000s 3,381 · 1990s 2,391 · 1980s 699 · 1970s 387 · 1960s 157 · 1950s 29 · 1940s 3 · 1930s 2 · 1920s 3, plus 2,527 tracks with no usable year; selecting 80s+90s gave a queue with 0 tracks outside them ("covers 3,090 tracks") |


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

The **Streaming** card in the web UI carries a *Stream through Icecast* checkbox:
it writes `icecast.enabled` and `Apply & restart Icecast` brings the relay up (or
down) without touching `.env`. The port field sets both the listen port and the
published port — in the bundled container the listen port must stay `8008`, so
change the host-side mapping with `ICECAST_PORT` in `.env` instead.

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
(English, `en_GB-alan-medium`, ~60 MB) into the volume; the DJ can speak out of
the box.
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

Three layers, in increasing precedence:

| Layer | Holds | Notes |
|---|---|---|
| `.env` | docker compose inputs: host ports, the host music path, Icecast credentials, worker counts | only read by `docker-compose.yml`; it maps them into the container's environment |
| container env (`VDJ_*`) | first-boot defaults the app reads directly | set by compose, or `docker run -e ...`; a value already saved in `config.json` wins over both |
| `data/config.json` | everything the web UI writes — the app's live settings | created on first run, never committed, survives restarts and upgrades |

So: change a host port or a mount in `.env`, change *behaviour* in the web UI (or
`config.json`). An **empty** value always means "unset": the bundled compose
forwards `"${VAR:-}"`, and a blank string neither clobbers a default nor crashes
the app (a blank numeric value used to raise `int("")` at import).

### `.env` (docker compose)

| Key | Default | Meaning |
|-----|---------|---------|
| `HTTP_PORT` | `8420` | Host port published for the UI + stream |
| `MUSIC_DIR` | `/mnt/mp3` | **Host** path to your music (mounted `:ro` at `/music`) |
| `VDJ_LOG_LEVEL` | `info` | Log verbosity (`debug`, `info`, `warning`, `error`) |
| `VDJ_NO_VOICE_DOWNLOAD` | `0` | `1` = skip the first-run voice download |
| `VDJ_ICECAST_ENABLED` | `1` | `0` disables Icecast delivery (the browser player still works) |
| `ICECAST_PORT` | `8008` | Icecast listen **and** published host port (both sides of the mapping — change together) |
| `ICECAST_MOUNT` | `virtualdj` | Mountpoint external players open (`http://<host>:<port>/<mount>`) |
| `ICECAST_SOURCE_PASSWORD` | `hackme` | Relay password the app's pusher uses to connect to Icecast |
| `ICECAST_PUBLIC_HOST` | _blank_ | Host shown in the web UI's stream URL (blank = derive from the browser) |
| `ICECAST_ADMIN_PASSWORD` | `admin` | Icecast admin/status password (loopback-only) |
| `ICECAST_RELAY_PASSWORD` | = source password | Icecast relay password (loopback-only) |
| `VDJ_ICECAST_HOSTNAME` | `virtual-dj` | Hostname Icecast reports in its status page (cosmetic) |
| `VDJ_MUSIC_DIR` | `/music` | Music folder **inside** the container (change with the volume mapping) |
| `VDJ_LOUDNESS_WORKERS` | `2` | Parallel loudness analyses (one ffmpeg each) |
| `VDJ_OLLAMA_URL` | _unset_ | Ollama endpoint (e.g. `http://host-gateway:11434` for host Ollama) |
| `VDJ_OLLAMA_MODEL` | _unset_ | Model used for DJ scripts |

### In-container environment overrides

Useful for `docker run` / a non-compose install; the web UI writes to
`config.json` instead:

| Variable | Default | Meaning |
|----------|---------|---------|
| `VDJ_DATA_DIR` | `./data` | Where state is kept (config, DB, voices, logs, cache) |
| `VDJ_DB_PATH` | `$VDJ_DATA_DIR/vdj.sqlite3` | Library database file |
| `VDJ_MUSIC_DIR` | `~/Music` | Initial music folder (the container image sets `/music`) |
| `VDJ_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `VDJ_OLLAMA_MODEL` | `qwen3.5:9b` | Model used for DJ scripts |
| `VDJ_HOST` / `VDJ_PORT` | `0.0.0.0` / `8420` | Bind address and port |
| `VDJ_LOG_LEVEL` | `info` | Console log verbosity |
| `VDJ_NO_VOICE_DOWNLOAD` | `0` | `1` = skip the first-run voice download |
| `VDJ_LOUDNESS_WORKERS` | `2` | Parallel loudness analyses |
| `VDJ_ICECAST_*` | see above | The Icecast settings compose forwards |

### `data/config.json`

The sections the web UI does not already document elsewhere in this file:

| Key | Default | Meaning |
|---|---|---|
| `music_dir` | `~/Music` (`/music` in the container) | Where the library lives |
| `voices_dir` | _blank_ = `$VDJ_DATA_DIR/voices` | Where Piper voice models are kept (point it at a bigger disk) |
| `playback.shuffle` | `true` | Flat shuffle when programs are off |
| `playback.genres` / `playback.decades` / `playback.artists` | `[]` | The *Genres/decades* card's selection: empty = everything |
| `playback.search` | `""` | Free-text filter applied to every queue fill |
| `playback.program.*` | see [Playlist rules](#playlist-rules) | Theming, program length, artist rules |
| `playback.program.disabled.{genre,artist,decade}` | `[]` | Themes switched off in the *Programs* card (a stored exclusion list, so a new genre starts ON) |
| `dj.enabled` | `true` | Speak between tracks at all |
| `dj.voice` | `en_GB-alan-medium` | Piper voice — also picks the DJ's language. Measured on one sentence: ~19 semitones of pitch movement, against ~7 for the previous default `en_US-amy-medium` (the flattest of 15 installed voices); `-high` models of the same voice measured no wider for twice the CPU |
| `dj.noise_w` | `0.8` | Rhythm variation (phoneme-width jitter). Piper's own default |
| `dj.sentence_silence` | `0.25` | Seconds of silence after each sentence — 0.25 gives the ~0.45 s gap of a real host, where the engine's own punctuation pauses only reach 0.1–0.26 s |
| `dj.style` | _warm radio host_ | The LLM persona |
| `dj.talk_min` / `dj.talk_max` | `2` / `4` | Tracks between DJ breaks (randomized) |
| `dj.sent_min` / `dj.sent_max` | `1` / `3` | Sentences per DJ line |
| `dj.speed` | `1.0` | Piper speaking rate |
| `dj.gain_db` | `0.0` | Trim applied to the voice only |
| `dj.noise_scale` | `0.667` | Piper expressiveness |
| `dj.prefetch_depth` | `3` | DJ lines rendered ahead of playback |
| `llm.enabled` / `llm.base_url` / `llm.model` | `true` / `127.0.0.1:11434` / `qwen3.5:9b` | The Ollama endpoint used for scripts |
| `llm.temperature` / `llm.retries` / `llm.timeout_s` | `0.7` / `2` / `120` | Sampling and resilience |
| `stream.bitrate_kbps` | `128` | MP3 bitrate of the broadcast |
| `stream.sample_rate` / `stream.channels` | `44100` / `2` | Encoder output format |
| `stream.station_name` | `Virtual DJ` | Shown by players and in Icecast |
| `icecast.enabled` | `false` | Stream through the bundled Icecast as well (the *Streaming* card) |
| `icecast.port` / `icecast.mount` / `icecast.public_host` | `8008` / `virtualdj` / _blank_ | Listen port, mountpoint, advertised host |
| `icecast.source_password` / `hostname` / `host` | `hackme` / `virtual-dj` / `127.0.0.1` | Credentials and identity |
| `enrich.enabled` | `true` | MusicBrainz/Wikipedia lookups for DJ facts |
| `enrich.timeout_s` / `enrich.user_agent` | `15` / `VirtualDJ/1.0 …` | Lookup politeness settings |
| `websearch.enabled` / `websearch.timeout_s` | `true` / `12` | Fallback web lookups for missing metadata |
| `ai.free_text_genre` | `true` | Let the local LLM fill genres web lookups miss |

`cache.*`, `logging.*` and `loudness.*` each have their own table below.


### Temporary data (cleaned up automatically)

The app renders a DJ line to an mp3 the first time it says it and caches it. That
cache, plus TTS scratch space and any partially written download, is **pruned
automatically** — you never have to clean it by hand:

| Knob (`data/config.json` → `cache`) | Default | Meaning |
|------------------------------------|---------|---------|
| `enabled` | `true` | Run the periodic sweep at all |
| `cleanup_interval_minutes` | `10` | How often the sweep runs |
| `dj_keep_files` | `100` | Keep at least this many recent breaks |
| `dj_max_age_hours` | `48` | Delete breaks older than this |
| `dj_max_mb` | `256` | Overall size budget for the break cache (`0` = no limit) |
| `tmp_grace_minutes` | `60` | Age before a partial `*.tmp`/`*.part` file is considered abandoned |

A sweep runs on startup and then on the interval. A break the queue is about to
play is never removed, and **voice models are never touched** (they are
downloads, not cache). The **Library** card shows the current footprint —
*"temporary data: N DJ breaks, X MB · keeps newest 100, sweeps every 10 min"* —
with a **Clean now** button that sweeps immediately.

### Log file

Besides the console (`docker logs`), the app writes a bounded rotating log to
`data/logs/virtual-dj.log` (`logging.to_file`, `logging.max_mb` = 5,
`logging.backups` = 3). That file is what to read when an incident needs
diagnosing without attaching to the container.

### Strict config writes

`PUT /api/config` rejects any key the app does not read with **HTTP 422** and
lists it (`{"unknown": ["playback.repeat_windows"]}`). A mistyped key used to be
stored and silently ignored — the UI showed the setting as saved while nothing
consumed it.

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
(`6,772 of 10,158 measured (67%)` on the reference library — your numbers will
differ — with a live bar, the ETA and the last analyzed file) and puts each track's measured gain next to it in the library list. Its
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
| `app/config.py` | `data/config.json` defaults + the env bridge (empty value = unset), path resolution (`DATA_DIR`, `voices_dir()`, `ensure_dirs`) |
| `app/db.py` | SQLite schema and queries: tracks, play history, loudness, misses |
| `app/library.py` | Filesystem scan + tag extraction (mutagen); **every queue-level rule lives here** — `query_tracks`, `interleave_artists` (cap + gap), `artist_positions`, `repeat_window`, `program_size_for`, the theme list |
| `app/enrich.py` | MusicBrainz / Wikipedia lookups (language-aware, misses cached) |
| `app/ai_meta.py` | LLM-assisted metadata repair (romanisation, missing genres) |
| `app/websearch.py` | Fallback web lookups for metadata the APIs miss |
| `app/dj.py` | Prompt building, Ollama call, Piper synthesis, voice catalogue |
| `app/voices.py` | Piper voice download / verification / listing |
| `app/scheduler.py` | The queue: themed programs, filtering, DJ cadence, DJ prefetch |
| `app/stream.py` | `Broadcaster` — ffmpeg encode, frame alignment, client fan-out, failure breaker |
| `app/loudness.py` | EBU R128 measurement, the worker pool, per-track gain |
| `app/icecast_server.py` | Renders `icecast.xml`, supervises the bundled Icecast2 |
| `app/icecast.py` / `app/icecast_metadata.py` | Relay pusher + now-playing metadata to Icecast |
| `app/maintenance.py` | Housekeeping: prunes the DJ cache, abandoned partials, bounded by `cache.*` |
| `app/textq.py` | Announcement text queue (what the DJ is about to say) |
| `app/server.py` | FastAPI: `/stream.mp3`, the REST API, WebSocket now-playing |
| `app/main.py` | Entry point (`python -m app.main`) |
| `web/` | Control panel (plain JS, no build step) |
| `tests/` | Pytest; the playlist rules have their own matrix script in the project skill |


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
.venv/bin/python -m pytest tests/ -q      # 320 tests, ~3.5 minutes
```

The suite covers tag parsing, DJ prompt/fact grounding, scheduler cadence and
prefetching, voice downloading, loudness measurement, the env→config contract
(a subprocess import with the exact environment compose produces), the
maintenance sweep (including what must never be pruned), and end-to-end
streaming against a real server — decoding the live stream with ffmpeg across
forced track switches to assert zero decode errors. Tests use temp directories
and never touch your library.

> Run it through the venv interpreter (as above) rather than a bare `pytest`:
> this repo shares a workspace with other projects, and a bare `pytest` can pick
> up their config and fail on collection.

## License

MIT
