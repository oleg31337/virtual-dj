/* Virtual DJ frontend. Plain JS, no build step. */
'use strict';

const $ = (id) => document.getElementById(id);
const api = async (path, options = {}) => {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.status === 204 ? null : res.json();
};

const state = {
  config: null,
  selected: new Set(),
  activeGenres: new Set(),
  tracks: [],
  playing: false,
};

const fmtTime = (s) => {
  if (!s && s !== 0) return '0:00';
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
};

/* ---------- audio player ---------- */

let audio = null;

function toggleListen() {
  if (state.playing) {
    if (audio) { audio.pause(); audio.src = ''; audio.load(); audio = null; }
    state.playing = false;
    $('listen').textContent = '▶ Listen';
    return;
  }
  // Cache-bust so the browser opens a fresh live connection every time.
  audio = new Audio(`/stream.mp3?t=${Date.now()}`);
  audio.volume = $('volume').value / 100;
  audio.play().then(() => {
    state.playing = true;
    $('listen').textContent = '⏹ Stop';
  }).catch((err) => {
    alert(`Could not start playback: ${err.message}`);
  });
}

/* ---------- live state ---------- */

function renderState(s) {
  $('np-kind').textContent = (s.kind || 'idle').toUpperCase();
  $('listeners').textContent = `${s.listeners} listener${s.listeners === 1 ? '' : 's'}`;
  const t = s.track;
  $('np-title').textContent = t ? (t.title || t.path.split('/').pop()) : '—';
  $('np-artist').textContent = t ? (t.artist || 'Unknown artist') : '';
  $('np-album').textContent = t ? [t.album, t.genre, t.year].filter(Boolean).join(' · ') : '';
  $('np-dj').textContent = s.dj_text || '—';
  const pct = s.duration && s.elapsed ? Math.min(100, (s.elapsed / s.duration) * 100) : 0;
  $('np-bar').style.width = `${pct}%`;
  $('np-time').textContent = s.duration
    ? `${fmtTime(s.elapsed)} / ${fmtTime(s.duration)}`
    : fmtTime(s.elapsed);
  $('pauseresume').textContent = s.paused ? '▶ Resume broadcast' : '⏸ Pause broadcast';
}

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'state') { renderState(msg.data); loadQueue(); }
  };
  ws.onclose = () => setTimeout(connectWS, 3000);
}

/* ---------- library ---------- */

async function loadTracks(search) {
  const q = new URLSearchParams({ limit: '300' });
  if (search) q.set('search', search);
  if (state.activeGenres.size) q.set('genre', [...state.activeGenres].join(','));
  state.tracks = await api(`/api/library/tracks?${q}`);
  const box = $('tracks');
  box.innerHTML = '';
  for (const tr of state.tracks) {
    const div = document.createElement('div');
    div.className = 'item' + (state.selected.has(tr.id) ? ' sel' : '');
    div.innerHTML =
      `<div class="t"><div>${esc(tr.title || tr.path.split('/').pop())}</div>` +
      `<div class="a">${esc(tr.artist || 'Unknown')}</div></div>` +
      `<div class="x">${fmtTime(tr.duration)}</div>`;
    div.onclick = () => {
      if (state.selected.has(tr.id)) state.selected.delete(tr.id);
      else state.selected.add(tr.id);
      div.classList.toggle('sel');
    };
    box.appendChild(div);
  }
}

const esc = (s) => String(s ?? '').replace(/[<>&"]/g,
  (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' }[c]));

async function loadGenres() {
  const genres = await api('/api/library/genres');
  const box = $('genres');
  box.innerHTML = '';
  for (const g of genres.slice(0, 120)) {
    const chip = document.createElement('div');
    chip.className = 'chip' + (state.activeGenres.has(g.genre) ? ' on' : '');
    chip.textContent = `${g.genre} (${g.n})`;
    chip.onclick = () => {
      if (state.activeGenres.has(g.genre)) state.activeGenres.delete(g.genre);
      else state.activeGenres.add(g.genre);
      chip.classList.toggle('on');
    };
    box.appendChild(chip);
  }
}

async function loadQueue() {
  const items = await api('/api/queue?limit=40');
  const box = $('queue');
  box.innerHTML = '';
  for (const it of items) {
    const div = document.createElement('div');
    div.className = 'item';
    div.innerHTML =
      `<div class="t"><div>${esc(it.track.title || it.track.path.split('/').pop())}` +
      `${it.dj_ready ? ' 🎙' : ''}</div>` +
      `<div class="a">${esc(it.track.artist || 'Unknown')}</div></div>` +
      `<div class="x">✕</div>`;
    div.querySelector('.x').onclick = async (e) => {
      e.stopPropagation();
      await api(`/api/queue/${it.uid}`, { method: 'DELETE' });
      loadQueue();
    };
    if (it.dj_text) div.title = it.dj_text;
    box.appendChild(div);
  }
}

async function loadHistory() {
  const rows = await api('/api/history?limit=30');
  $('history').innerHTML = rows.map((r) =>
    `<div class="item"><div class="t"><div>${esc(r.title || '—')}</div>` +
    `<div class="a">${esc(r.artist || '')}</div></div>` +
    `<div class="x">${new Date(r.played_at * 1000).toLocaleTimeString()}</div></div>`
  ).join('');
}

async function loadPresets() {
  const presets = await api('/api/presets');
  const box = $('presets');
  box.innerHTML = '';
  for (const p of presets) {
    const div = document.createElement('div');
    div.className = 'item';
    div.innerHTML = `<div class="t">${esc(p.name)}</div><div class="x">apply ✕</div>`;
    div.querySelector('.t').onclick = async () => {
      await api(`/api/presets/${encodeURIComponent(p.name)}/apply`, { method: 'POST' });
      await loadConfig(); loadQueue();
    };
    div.querySelector('.x').onclick = async (e) => {
      e.stopPropagation();
      if (e.target.textContent.includes('✕')) {
        await api(`/api/presets/${encodeURIComponent(p.name)}`, { method: 'DELETE' });
        loadPresets();
      }
    };
    box.appendChild(div);
  }
}

/* ---------- config ---------- */

async function loadConfig() {
  const cfg = await api('/api/config');
  state.config = cfg;
  $('station-name').textContent = cfg.stream.station_name;
  $('music-dir').value = cfg.music_dir;
  $('dj-freq').value = cfg.dj.every_n_tracks;
  $('freq-val').textContent = cfg.dj.every_n_tracks || 'never';
  $('dj-sentences').value = cfg.dj.max_sentences;
  $('sent-val').textContent = cfg.dj.max_sentences;
  $('dj-speed').value = Math.round(cfg.dj.speed * 100);
  $('speed-val').textContent = cfg.dj.speed.toFixed(2);
  $('dj-style').value = cfg.dj.style;
  $('dj-enabled').checked = cfg.dj.enabled;
  $('enrich-enabled').checked = cfg.enrich.enabled;
  $('shuffle').checked = cfg.playback.shuffle;
  state.activeGenres = new Set(cfg.playback.genres || []);

  const voices = await api('/api/dj/voices');
  $('dj-voice').innerHTML = voices.voices
    .map((v) => `<option${v === voices.current ? ' selected' : ''}>${esc(v)}</option>`)
    .join('') || '<option>no voices installed</option>';
}

async function loadHealth() {
  try {
    const h = await api('/api/health');
    const bits = [
      `${h.library.total} tracks`,
      h.llm.ok ? `LLM ok (${state.config?.llm?.model || ''})` : 'LLM offline',
      h.tts.ok ? 'voice ok' : 'voice missing',
    ];
    $('health-line').innerHTML =
      `<span class="${h.llm.ok && h.tts.ok ? 'ok' : 'bad'}">${esc(bits.join(' · '))}</span>`;
  } catch (e) {
    $('health-line').textContent = 'backend unreachable';
  }
}

async function loadLibraryStats() {
  try {
    const s = await api('/api/library');
    const stats = s.library;
    const sources = stats.meta_sources || {};
    const sourceBits = Object.entries(sources)
      .map(([k, v]) => `${v} from ${k}`).join(', ');
    let html = `playable <b>${stats.playable}</b> / ${stats.total} total`;
    if (stats.excluded)
      html += ` · <span class="warn">${stats.excluded} skipped</span>`;
    if (sourceBits) html += `<br><span class="dim">guessed: ${esc(sourceBits)}</span>`;
    if (stats.excluded && stats.unknown_reasons) {
      const reasons = Object.entries(stats.unknown_reasons || {})
        .map(([k, v]) => `${esc(stats.reason_labels?.[k] || k)}: ${v}`)
        .join(', ');
      if (reasons)
        html += `<br><span class="warn">unknown: ${reasons}</span>`;
    }
    $('library-stats').innerHTML = html;
  } catch (e) { /* ignore */ }
}

async function pollScan() {
  const s = await api('/api/library/scan');
  $('scan-status').textContent = s.running
    ? `scanning… ${s.total_seen} files seen, ${s.added} new`
    : (s.error ? `error: ${s.error}`
       : (s.finished_at ? `last scan: +${s.added} new, ${s.updated} updated` : 'idle'));
  if (s.running) setTimeout(pollScan, 1500);
  else { loadGenres(); loadHealth(); loadLibraryStats(); }
}

/* ---------- wiring ---------- */

function wire() {
  $('listen').onclick = toggleListen;
  $('volume').oninput = (e) => { if (audio) audio.volume = e.target.value / 100; };
  $('stream-link').href = `${location.origin}/stream.mp3`;

  $('skip').onclick = () => api('/api/transport/skip', { method: 'POST' });
  $('pauseresume').onclick = async () => {
    const paused = $('pauseresume').textContent.includes('Resume');
    await api(`/api/transport/${paused ? 'resume' : 'pause'}`, { method: 'POST' });
  };
  $('preview').onclick = async () => {
    $('preview').disabled = true;
    $('preview').textContent = '🎙 generating…';
    try {
      const r = await api('/api/dj/preview', { method: 'POST', body: '{}' });
      $('np-dj').textContent = r.text;
      if (r.audio_url) {
        const el = $('preview-audio');
        el.hidden = false; el.src = r.audio_url; el.play();
      }
    } catch (e) { alert(e.message); }
    $('preview').disabled = false;
    $('preview').textContent = '🎙 Preview DJ line';
  };

  $('save-dir').onclick = async () => {
    await api('/api/config', {
      method: 'PUT', body: JSON.stringify({ music_dir: $('music-dir').value.trim() }),
    });
    loadConfig();
  };
  $('scan').onclick = async () => {
    await api('/api/library/scan', {
      method: 'POST',
      body: JSON.stringify({ music_dir: $('music-dir').value.trim() }),
    });
    pollScan();
  };
  $('do-search').onclick = () => loadTracks($('search').value.trim());
  $('search').onkeydown = (e) => { if (e.key === 'Enter') $('do-search').click(); };

  $('queue-selected').onclick = async () => {
    if (!state.selected.size) return;
    await api('/api/queue', {
      method: 'POST',
      body: JSON.stringify({ track_ids: [...state.selected] }),
    });
    state.selected.clear(); loadTracks($('search').value.trim()); loadQueue();
  };
  $('play-selected').onclick = async () => {
    if (!state.selected.size) return;
    await api('/api/queue', {
      method: 'POST',
      body: JSON.stringify({ track_ids: [...state.selected], replace: true }),
    });
    await api('/api/transport/skip', { method: 'POST' });
    state.selected.clear(); loadTracks($('search').value.trim()); loadQueue();
  };
  $('clear-queue').onclick = async () => {
    await api('/api/queue/clear', { method: 'POST' }); loadQueue();
  };

  $('apply-filters').onclick = async () => {
    await api('/api/config', {
      method: 'PUT',
      body: JSON.stringify({
        playback: {
          genres: [...state.activeGenres],
          shuffle: $('shuffle').checked,
          search: $('search').value.trim(),
        },
      }),
    });
    await api('/api/queue/clear', { method: 'POST' });
    loadTracks($('search').value.trim()); loadQueue();
  };

  $('dj-freq').oninput = (e) =>
    ($('freq-val').textContent = e.target.value === '0' ? 'never' : e.target.value);
  $('dj-sentences').oninput = (e) => ($('sent-val').textContent = e.target.value);
  $('dj-speed').oninput = (e) =>
    ($('speed-val').textContent = (e.target.value / 100).toFixed(2));

  $('save-dj').onclick = async () => {
    await api('/api/config', {
      method: 'PUT',
      body: JSON.stringify({
        dj: {
          enabled: $('dj-enabled').checked,
          every_n_tracks: Number($('dj-freq').value),
          max_sentences: Number($('dj-sentences').value),
          speed: Number($('dj-speed').value) / 100,
          voice: $('dj-voice').value,
          style: $('dj-style').value.trim(),
        },
        enrich: { enabled: $('enrich-enabled').checked },
      }),
    });
    loadConfig();
  };

  $('save-preset').onclick = async () => {
    const name = $('preset-name').value.trim();
    if (!name) return;
    await api('/api/presets', { method: 'POST', body: JSON.stringify({ name }) });
    $('preset-name').value = ''; loadPresets();
  };
}

async function init() {
  wire();
  await loadConfig();
  await Promise.all([loadGenres(), loadTracks(''), loadQueue(),
                     loadPresets(), loadHistory(), loadHealth()]);
  pollScan();
  connectWS();
  setInterval(loadHistory, 30000);
  setInterval(loadHealth, 30000);
}

init();
