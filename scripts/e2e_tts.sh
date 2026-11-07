#!/usr/bin/env bash
# End-to-end check of the station + TTS-service split.
#
# Runs against a live stack (docker compose up -d) and exercises everything that
# would leave the DJ silent while the API looks fine:
#
#   1. the service's health and capability document (engine, voices, steps)
#   2. a real render through the service (Russian, 5 steps, params honoured)
#   3. the app's discovery of that document          GET /api/dj/tts
#   4. a real preview render through the app         POST /api/dj/preview
#      (text shaping + loudnorm + mp3 mastering)
#   5. the app's health payload naming engine/voice/language
#   6. the audio the app produced (ffprobe)
#
# Usage: scripts/e2e_tts.sh [app_base] [tts_base]
#   defaults: http://127.0.0.1:8420  http://127.0.0.1:9000
#   a second compose project: scripts/e2e_tts.sh http://127.0.0.1:18420 http://127.0.0.1:19000
# Artifacts land in $E2E_OUT (default: a fresh temp dir), printed at the end.
# Exit code = number of failed checks.
set -uo pipefail
APP="${1:-http://127.0.0.1:8420}"
TTS="${2:-http://127.0.0.1:9000}"
OUT="${E2E_OUT:-$(mktemp -d -t vdj-e2e-XXXXXX)}"
mkdir -p "$OUT"
FAIL=0

say() { printf '\n=== %s\n' "$*"; }
ok()  { printf '  PASS  %s\n' "$*"; }
bad() { printf '  FAIL  %s\n' "$*"; FAIL=$((FAIL + 1)); }

command -v ffprobe >/dev/null || { echo "ffprobe is required"; exit 1; }
python3 -c 'import json' || { echo "python3 is required"; exit 1; }

say "1. service health"
curl -sf -m 20 "$TTS/v1/health" -o "$OUT/health.json" && python3 -c "
import json
d=json.load(open('$OUT/health.json'))
assert d.get('ok') is True, d
print('  engine:', d.get('engine'))
print('  model_ready:', d.get('model_ready'), '| voices_ready:', d.get('voices_ready'))
print('  steps:', d.get('steps'), '| renders:', d.get('renders'), '| uptime_s:', d.get('uptime_s'))
" && ok "health ok" || bad "health"

say "2. capability document"
curl -sf -m 20 "$TTS/v1/capabilities" -o "$OUT/caps.json" && python3 -c "
import json
d=json.load(open('$OUT/caps.json'))
assert d['quality']['steps']==5, d['quality']
voices={v['id']:v for v in d['voices']}
assert voices, 'no voices'
langs=set()
for v in voices.values(): langs.update(v['languages'])
assert {'en','ru'} <= langs, langs
print('  engine:', d['engine']['id'], d['engine'].get('version'))
print('  voices:', ', '.join(f\"{k}{'*' if v['installed'] else ''}\" for k,v in voices.items()))
print('  languages:', sorted(langs))
print('  params:', [(p['key'], p.get('min'), p.get('max'), p.get('default')) for p in d['params']])
print('  steps:', d['quality']['steps'])
" && ok "capabilities ok" || bad "capabilities"

say "3. service render (ru, params honoured, unknown param ignored)"
curl -sS -m 300 -D "$OUT/speech.hdr" -o "$OUT/svc.wav" -X POST "$TTS/v1/speech" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Привет, это проверка службы синтеза речи номер один.","voice":null,"language":"ru","params":{"speed":1.0,"sentence_pause":0.3,"nonsense":1}}'
python3 -c "
import re
h=open('$OUT/speech.hdr').read()
def g(k):
    m=re.search(rf'^{k}:\s*(.+)\$', h, re.M|re.I)
    return m.group(1).strip() if m else None
steps=g('x-tts-steps'); dur=g('x-tts-duration'); ignored=g('x-tts-params-ignored')
print('  steps:', steps, '| duration:', dur, '| ignored:', ignored)
assert steps=='5', steps
assert float(dur or 0) > 1.0, dur
assert 'nonsense' in (ignored or ''), ignored
" && ok "service render ok" || bad "service render"
ffprobe -v error -show_entries format=duration,size -of default=nw=1 "$OUT/svc.wav" | sed 's/^/  /'

say "4. app sees the same document"
curl -sf -m 30 "$APP/api/dj/tts" -o "$OUT/app_tts.json" && python3 -c "
import json
d=json.load(open('$OUT/app_tts.json'))
assert d['ok'] is True, d
assert d['problem'] is None, d['problem']
ids={v['id'] for v in d['voices']}
print('  engine:', d['engine']['id'], '| current:', d['current'])
print('  params:', [p['key'] for p in d['params']], '| installed:', len(d['installed']))
assert d['current']['voice'] in ids, d['current']
assert d['current']['voice'] in d['installed'], d['installed']
assert d['quality']['steps']==5
" && ok "app discovery ok" || bad "app discovery"

say "5. app preview render (text shaping + mastering)"
curl -sf -m 300 -X POST "$APP/api/dj/preview" -H 'Content-Type: application/json' \
  -d '{"text":"Сегодня в эфире запись 1989 года, темп 120 BPM.","language":"ru","params":{"speed":1.05,"sentence_pause":0.4}}' \
  -o "$OUT/preview.json" && python3 -c "
import json
d=json.load(open('$OUT/preview.json'))
print('  text:', d.get('text'))
print('  audio:', d.get('audio_url'), '| duration:', d.get('duration'))
t=d.get('text') or ''
assert '1989' not in t, t
assert 'тысяча девятьсот восемьдесят' in t.lower(), t
assert '120' not in t, t
" && ok "preview text shaped" || bad "preview text"

say "6. app health"
curl -sf -m 20 "$APP/api/health" -o "$OUT/health_app.json" && python3 -c "
import json
d=json.load(open('$OUT/health_app.json'))['tts']
print('  ok:', d['ok'], '| engine:', (d.get('engine') or {}).get('id'),
      '| voice:', d.get('voice'), '| language:', d.get('language'))
print('  params:', d.get('params'), '| base_url:', d.get('base_url'))
assert d['ok'] is True, d
assert (d.get('engine') or {}).get('id'), d
assert d.get('language'), d
" && ok "app health ok" || bad "app health"

say "7. the audio the app produced"
audio=$(python3 -c "
import json
print(json.load(open('$OUT/preview.json')).get('audio_url') or '')
")
if [ -n "$audio" ]; then
  case "$audio" in
    http*) url="$audio" ;;
    /*) url="$APP$audio" ;;
    *) url="$APP/$audio" ;;
  esac
  if curl -sf -m 120 "$url" -o "$OUT/preview.mp3"; then
    ffprobe -v error -show_entries stream=codec_name,sample_rate,channels \
      -show_entries format=duration -of default=nw=1 "$OUT/preview.mp3" | sed 's/^/  /'
    ok "preview audio fetched"
  else
    bad "could not fetch $url"
  fi
else
  bad "preview returned no audio path"
fi

say "RESULT"
printf '  artifacts: %s\n' "$OUT"
if [ "$FAIL" -eq 0 ]; then echo "  ALL CHECKS PASSED"; else echo "  $FAIL CHECK(S) FAILED"; fi
exit "$FAIL"
