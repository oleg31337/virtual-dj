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
  activeLanguages: new Set(),
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
  const prog = s.program;
  if (prog && prog.label) {
    const word = { genre: 'genre', artist: 'artist', decade: 'era', language: 'language' }[prog.kind] || 'set';
    $('np-program').textContent = `ON AIR · ${prog.label} (${word})`;
    $('np-program').style.display = '';
  } else {
    $('np-program').style.display = 'none';
  }
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
  if (state.activeLanguages.size) q.set('language', [...state.activeLanguages].join(','));
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

async function loadLanguages() {
  const languages = await api('/api/library/languages');
  const box = $('languages');
  box.innerHTML = '';
  for (const l of languages.slice(0, 50)) {
    const chip = document.createElement('div');
    chip.className = 'chip' + (state.activeLanguages.has(l.language) ? ' on' : '');
    chip.textContent = `${l.language} (${l.n})`;
    chip.onclick = () => {
      if (state.activeLanguages.has(l.language)) state.activeLanguages.delete(l.language);
      else state.activeLanguages.add(l.language);
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
  const tmin = cfg.dj.talk_min ?? 2, tmax = cfg.dj.talk_max ?? 4;
  $('dj-talk-min').value = tmin;
  $('dj-talk-max').value = tmax;
  $('talk-val').textContent = (tmax === 0) ? 'never' : `${tmin}–${tmax}`;
  const smin = cfg.dj.sent_min ?? 1, smax = cfg.dj.sent_max ?? 3;
  $('dj-sent-min').value = smin;
  $('dj-sent-max').value = smax;
  $('sent-val').textContent = `${smin}–${smax}`;
  $('dj-speed').value = Math.round(cfg.dj.speed * 100);
  $('speed-val').textContent = cfg.dj.speed.toFixed(2);
  $('dj-style').value = cfg.dj.style;
  $('dj-enabled').checked = cfg.dj.enabled;
  $('enrich-enabled').checked = cfg.enrich.enabled;
  $('shuffle').checked = cfg.playback.shuffle;
  state.activeGenres = new Set(cfg.playback.genres || []);
  state.activeLanguages = new Set(cfg.playback.languages || []);

  const voices = await api('/api/dj/voices');
  const profiles = (voices.profiles || []).filter((p) => p.lang === 'english');
  $('dj-voice').innerHTML = profiles.length
    ? profiles.map((p) =>
        `<option value="${esc(p.id)}"${p.id === voices.current ? ' selected' : ''}>${esc(p.name)} (${esc(p.gender)})${p.installed ? '' : ' — not installed'}</option>`).join('')
    : (voices.voices.length
        ? voices.voices.map((v) => `<option${v === voices.current ? ' selected' : ''}>${esc(v)}</option>`).join('')
        : '<option>no voices installed</option>');
  // Show the intonation note for the selected voice.
  const sel = profiles.find((p) => p.id === $('dj-voice').value) || profiles[0];
  $('dj-voice-note').textContent = sel && sel.note ? sel.note : '';

  // Russian voice picker (only Russian-language tracks use it).
  const ru = voices.russian_profiles || [];
  $('dj-voice-ru').innerHTML = ru.length
    ? ru.map((p) =>
        `<option value="${esc(p.id)}"${p.id === voices.current_russian ? ' selected' : ''}>${esc(p.name)} (${esc(p.gender)})${p.installed ? '' : ' — not installed'}</option>`).join('')
    : '<option value="">no Russian voices installed</option>';
  const rusel = ru.find((p) => p.id === $('dj-voice-ru').value) || ru[0];
  $('dj-voice-ru-note').textContent = rusel && rusel.note ? rusel.note : '';

  // Program grouping settings.
  $('program-enabled').checked = !!(cfg.playback?.program?.enabled ?? true);
  $('program-size').value = cfg.playback?.program?.size ?? 6;
  $('program-size-val').textContent = $('program-size').value;
  $('program-strategy-sel').value = cfg.playback?.program?.strategy ?? 'genre';
  // Voice + prosody controls.
  $('dj-speed').value = Math.round((cfg.dj?.speed ?? 1.0) * 100);
  $('speed-val').textContent = ((cfg.dj?.speed ?? 1.0)).toFixed(2);
  const ns = cfg.dj?.noise_scale ?? 0.667;
  $('dj-noise').value = Math.round(ns * 100);
  $('expr-val').textContent = ns.toFixed(2);
}

async function loadLLMConfig() {
  const cfg = state.config;
  if (!cfg || !cfg.llm) return;
  $('llm-enabled').checked = !!(cfg.llm.enabled ?? true);
  $('llm-url').value = cfg.llm.base_url || '';
  $('llm-timeout').value = cfg.llm.timeout_s ?? 120;
  $('llm-temperature').value = cfg.llm.temperature ?? 0.7;
  $('llm-retries').value = cfg.llm.retries ?? 2;
  // Pre-populate the dropdown with the configured model (no probe yet).
  const cur = cfg.llm.model || '';
  $('llm-model').innerHTML = cur
    ? `<option value="${esc(cur)}" selected>${esc(cur)}</option>`
    : '<option value="">— set URL and click Load models —</option>';
  await loadLLMStatus();
}

async function loadLLMStatus() {
  try {
    const h = await api('/api/health');
    const llm = h.llm || {};
    const el = $('llm-status');
    if (llm.ok) {
      el.textContent = `connected · ${llm.models?.length || 0} models`
        + (llm.model_present ? ' · configured model present' : ' · configured model MISSING');
      el.className = 'meta ' + (llm.model_present ? 'ok' : 'warn');
    } else {
      el.textContent = `not connected: ${llm.error || 'no base_url configured'}`;
      el.className = 'meta warn';
    }
  } catch (e) { /* ignore */ }
}

async function loadLLMModels() {
  const url = $('llm-url').value.trim();
  const btn = $('llm-load-models');
  btn.disabled = true; btn.textContent = 'loading…';
  try {
    const r = await api('/api/llm/models', {
      method: 'POST',
      body: JSON.stringify({ base_url: url || undefined }),
    });
    const sel = $('llm-model');
    if (r.ok && r.models && r.models.length) {
      const cur = sel.value;
      sel.innerHTML = r.models
        .map((m) => `<option value="${esc(m)}"${m === cur ? ' selected' : ''}>${esc(m)}</option>`)
        .join('');
      $('llm-status').textContent = `${r.models.length} models found`;
      $('llm-status').className = 'meta ok';
    } else {
      sel.innerHTML = '<option value="">— no models / unreachable —</option>';
      $('llm-status').textContent = `no models: ${r.error || 'unknown'}`;
      $('llm-status').className = 'meta warn';
    }
  } catch (e) {
    $('llm-status').textContent = `error: ${e.message}`;
    $('llm-status').className = 'meta warn';
  } finally {
    btn.disabled = false; btn.textContent = 'Load models';
  }
}

async function testLLM() {
  const url = $('llm-url').value.trim();
  const model = $('llm-model').value;
  const btn = $('llm-test');
  const out = $('llm-test-result');
  btn.disabled = true; btn.textContent = 'testing…';
  out.textContent = 'contacting model…';
  out.className = 'meta dim';
  try {
    const r = await api('/api/llm/test', {
      method: 'POST',
      body: JSON.stringify({ base_url: url || undefined, model: model || undefined }),
    });
    if (r.ok && r.text) {
      out.textContent = `✓ ${r.model}: "${r.text.trim()}"`;
      out.className = 'meta ok';
    } else {
      out.textContent = `✗ ${r.error || 'no response'}`;
      out.className = 'meta warn';
    }
  } catch (e) {
    out.textContent = `✗ ${e.message}`;
    out.className = 'meta warn';
  } finally {
    btn.disabled = false; btn.textContent = 'Test connection';
  }
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

async function loadPrograms() {
  try {
    const p = await api('/api/programs');
    $('program-strategy').textContent =
      `(${{ genre: 'by genre', artist: 'by artist', decade: 'by decade', language: 'by language' }[p.strategy] || p.strategy})`;
    const themes = (p.themes || []).slice(0, 12);
    if (!themes.length) {
      $('programs').innerHTML = '<div class="dim">No themes available yet.</div>';
      return;
    }
    $('programs').innerHTML = themes.map((t) => {
      const kind = p.strategy === 'artist' ? 'artist'
                 : p.strategy === 'decade' ? 'decade'
                 : p.strategy === 'language' ? 'language' : 'genre';
      let label;
      if (p.strategy === 'decade') {
        label = `${t.decade}s`;
      } else if (p.strategy === 'language') {
        label = t.language || '?';
      } else {
        label = t.genre || t.artist || t.label || '?';
      }
      return `<div class="program-chip" data-kind="${esc(kind)}">`
        + `<span class="pc-label">${esc(label)}</span>`
        + `<span class="pc-n">${t.n}</span></div>`;
    }).join('');
  } catch (e) { /* ignore */ }
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
  const el = $('scan-status');
  if (s.running) {
    el.textContent = `scanning… ${s.total_seen} files seen, ${s.added} new`;
    el.className = 'meta dim';
    setTimeout(pollScan, 1500);
  } else if (s.error) {
    el.textContent = `⚠ ${s.error}`;
    el.className = 'meta warn';
    loadHealth();
  } else {
    el.textContent = s.finished_at
      ? `last scan: +${s.added} new, ${s.updated} updated`
      : 'idle';
    el.className = 'meta dim';
    loadGenres(); loadLanguages(); loadHealth(); loadLibraryStats();
  }
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
  $('llm-load-models').onclick = () => loadLLMModels();
  $('llm-test').onclick = () => testLLM();
  $('save-llm').onclick = async () => {
    const patch = {
      llm: {
        enabled: $('llm-enabled').checked,
        base_url: $('llm-url').value.trim(),
        model: $('llm-model').value,
        timeout_s: Number($('llm-timeout').value) || 120,
        temperature: Number($('llm-temperature').value) || 0.7,
        retries: Number($('llm-retries').value) || 0,
      },
    };
    await api('/api/config', { method: 'PUT', body: JSON.stringify(patch) });
    await loadLLMStatus();
    loadHealth();
  };
  $('llm-url').onchange = () => { $('llm-model').innerHTML = '<option value="">— click Load models —</option>'; };
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

  $('apply-language-filters').onclick = async () => {
    await api('/api/config', {
      method: 'PUT',
      body: JSON.stringify({
        playback: { languages: [...state.activeLanguages] },
      }),
    });
    await api('/api/queue/clear', { method: 'POST' });
    loadTracks($('search').value.trim()); loadQueue();
  };

  $('dj-speed').oninput = (e) =>
    ($('speed-val').textContent = (e.target.value / 100).toFixed(2));

  // Selecting a voice (English or Russian) downloads it on demand if missing,
  // verifies it actually landed on disk, refreshes the picker, then selects AND
  // activates it (so the DJ starts using it). This is the full expected flow.
  const selectVoice = async (selectId, noteId, langKey) => {
    const voice = $(selectId).value;
    const note = $(noteId);
    const data = await api('/api/dj/voices');
    const installed = data.voices || [];
    if (!installed.includes(voice)) {
      note.textContent = '⏳ downloading…';
      note.className = 'dim';
      let r;
      try {
        r = await api('/api/dj/voices/download', {
          method: 'POST',
          body: JSON.stringify({ voice }),
        });
      } catch (e) {
        note.textContent = `✗ ${e.message}`;
        note.className = 'warn';
        return;
      }
      // "Verify it is ok": the download must report success AND the file must
      // now appear in the available (on-disk) list.
      const ok = (r.downloaded || []).includes(voice)
                 && (r.available || []).includes(voice);
      if (!ok) {
        note.textContent = `✗ download failed: ${(r.failed || []).join(', ') || 'unknown'}`;
        note.className = 'warn';
        return;
      }
      note.textContent = '✓ downloaded & verified';
    } else {
      note.textContent = '✓ installed';
    }
    note.className = 'dim';
    // Refresh the picker choices, then select + activate the chosen voice.
    await loadConfig();
    $(selectId).value = voice;
    const profiles = (langKey === 'russian' ? data.russian_profiles : data.profiles) || [];
    const sel = profiles.find((p) => p.id === voice);
    if (sel && sel.note) $(noteId).textContent = sel.note;
    // Persist as the active voice for this language so the DJ uses it.
    const patch = langKey === 'russian'
      ? { dj: { russian_voice: voice } }
      : { dj: { voice } };
    try {
      await api('/api/config', { method: 'PUT', body: JSON.stringify(patch) });
    } catch (e) {
      // Selection still active in the UI even if the save round-trip fails.
    }
    // Update the "current" highlight on the next catalogue read.
    void loadConfig();
  };
  $('dj-voice').onchange = () => selectVoice('dj-voice', 'dj-voice-note', 'english');
  $('dj-voice-ru').onchange = () => selectVoice('dj-voice-ru', 'dj-voice-ru-note', 'russian');
  $('test-voice-ru').onclick = async () => {
    const voice = $('dj-voice-ru').value;
    const speed = Number($('dj-speed').value) / 100;
    const noise = Number($('dj-noise').value) / 100;
    const btn = $('test-voice-ru');
    btn.disabled = true;
    btn.textContent = '⏳ synthesizing…';
    try {
      const r = await api('/api/dj/preview', {
        method: 'POST',
        body: JSON.stringify({
          text: 'В эфире Виртуальный DJ. Сейчас прозвучит трек, '
                + 'который вы давно ждали.',
          voice, language: 'russian', speed, noise_scale: noise,
        }),
      });
      if (r && r.audio_url) {
        const a = $('preview-audio');
        a.hidden = false;
        a.src = r.audio_url;
        a.play().catch(() => {});
      }
    } catch (e) {
      alert('Voice test failed: ' + e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = '▶ Test Russian voice';
    }
  };

  $('save-program').onclick = async () => {
    await api('/api/config', {
      method: 'PUT',
      body: JSON.stringify({
        playback: {
          program: {
            enabled: $('program-enabled').checked,
            size: Number($('program-size').value),
            strategy: $('program-strategy-sel').value,
          },
        },
      }),
    });
    await loadPrograms();
  };
  $('program-size').oninput = (e) =>
    ($('program-size-val').textContent = e.target.value);
  $('save-dj').onclick = async () => {
    let tmin = Number($('dj-talk-min').value), tmax = Number($('dj-talk-max').value);
    if (tmax < tmin) [tmin, tmax] = [tmax, tmin];
    let smin = Number($('dj-sent-min').value), smax = Number($('dj-sent-max').value);
    if (smax < smin) [smin, smax] = [smax, smin];
    await api('/api/config', {
      method: 'PUT',
      body: JSON.stringify({
        dj: {
          enabled: $('dj-enabled').checked,
          talk_min: tmin,
          talk_max: tmax,
          sent_min: smin,
          sent_max: smax,
          speed: Number($('dj-speed').value) / 100,
          noise_scale: Number($('dj-noise').value) / 100,
          voice: $('dj-voice').value,
          style: $('dj-style').value.trim(),
          russian_voice: $('dj-voice-ru').value || undefined,
        },
        enrich: { enabled: $('enrich-enabled').checked },
      }),
    });
    loadConfig();
  };
  const refreshTalkLabel = () => {
    const tmin = Number($('dj-talk-min').value), tmax = Number($('dj-talk-max').value);
    $('talk-val').textContent = (tmax === 0 && tmin === 0) ? 'never'
      : `${Math.min(tmin, tmax)}–${Math.max(tmin, tmax)}`;
  };
  const refreshSentLabel = () => {
    const smin = Number($('dj-sent-min').value), smax = Number($('dj-sent-max').value);
    $('sent-val').textContent = `${Math.min(smin, smax)}–${Math.max(smin, smax)}`;
  };
  $('dj-talk-min').oninput = refreshTalkLabel;
  $('dj-talk-max').oninput = refreshTalkLabel;
  $('dj-sent-min').oninput = refreshSentLabel;
  $('dj-sent-max').oninput = refreshSentLabel;
  $('dj-noise').oninput = (e) =>
    ($('expr-val').textContent = (e.target.value / 100).toFixed(2));
  $('test-voice').onclick = async () => {
    const voice = $('dj-voice').value;
    const speed = Number($('dj-speed').value) / 100;
    const noise = Number($('dj-noise').value) / 100;
    const btn = $('test-voice');
    btn.disabled = true;
    btn.textContent = '⏳ synthesizing…';
    try {
      const r = await api('/api/dj/preview', {
        method: 'POST',
        body: JSON.stringify({
          text: 'Hey listeners, this is your Virtual DJ — let\'s keep the '
                + 'vibes flowing through the night.',
          voice, speed, noise_scale: noise,
        }),
      });
      if (r && r.audio_url) {
        const a = $('preview-audio');
        a.hidden = false;
        a.src = r.audio_url;
        a.play().catch(() => {});
      }
    } catch (e) {
      alert('Voice test failed: ' + e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = '▶ Test this voice';
    }
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
  await Promise.all([loadGenres(), loadLanguages(), loadTracks(''), loadQueue(),
                     loadPresets(), loadHistory(), loadHealth(), loadPrograms(),
                     loadLLMConfig()]);
  pollScan();
  connectWS();
  setInterval(loadHistory, 30000);
  setInterval(loadHealth, 30000);
}

init();
