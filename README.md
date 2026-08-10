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
  on the web** (MusicBrainz + iTunes, fully cached). Tracks that are still
  unidentifiable, or written in a script the DJ can't read aloud, are skipped
  from playlists and reported as `unknown:` in the Library panel — so the DJ
  only ever introduces music it can name correctly.
- **Web control panel** — live now-playing, library browser/search, genre
  filters, queue editing, presets, and DJ settings. Includes a browser player.
- **Your library, untouched** — the scanner only ever reads your music files.

## Requirements

- Linux, Python 3.11+
- `ffmpeg` and `ffprobe` on PATH (`sudo apt install ffmpeg`)
- [Ollama](https://ollama.com) reachable on your network, for DJ scripts
  (optional — the station plays music fine without it)

## Install

```bash
git clone <your-repo-url> virtual-dj
cd virtual-dj
./run.sh                      # creates .venv, installs deps, starts on :8420
```

Then open **http://localhost:8420** and set your music folder in the Library
panel, or pre-seed it:

```bash
cp config.example.json data/config.json   # then edit
```

### The DJ voice

The Piper voice model is downloaded once into `data/voices/`:

```bash
mkdir -p data/voices && cd data/voices
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
```

Any Piper voice works — drop the `.onnx` + `.onnx.json` pair in the same folder
and pick it in the DJ Settings panel.

## Listening

| Player  | How |
|---------|-----|
| VLC     | Media → Open Network Stream → `http://<host>:8420/stream.mp3` |
| Winamp  | File → Play URL → same URL |
| Sonos   | Add a radio station by URL in the S2 app |
| Browser | Just open `http://<host>:8420` and hit **Listen** |
| CLI     | `mpv http://<host>:8420/stream.mp3` |

## Configuration

Settings live in `data/config.json` (created on first run, never committed).
The web UI writes this same file — you rarely need to edit it by hand.

A few env vars override the defaults at first boot, useful for containers:

| Variable | Default | Meaning |
|----------|---------|---------|
| `VDJ_MUSIC_DIR` | `~/Music` | Initial music folder |
| `VDJ_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `VDJ_OLLAMA_MODEL` | `qwen3.5:9b` | Model used for DJ scripts |
| `VDJ_DATA_DIR` | `./data` | Where state is kept |
| `VDJ_LOG_LEVEL` | `info` | Log verbosity |

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
.venv/bin/python -m pytest tests/ -q     # 89 tests
```

The suite covers tag parsing, DJ prompt/fact grounding, scheduler cadence, and
end-to-end streaming against a real server — including decoding the live stream
with ffmpeg across forced track switches to assert zero decode errors. Tests use
temp directories and never touch your library.

> Note for `hermes verify` / `uv` users: activate the venv first
> (`source .venv/bin/activate`), since the detected recipe invokes bare
> `pytest` and `uvicorn`.

## License

MIT
