"""HTTP API tests using FastAPI's TestClient against the real app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import config, db, library


@pytest.fixture
def client(music_dir, has_ffmpeg, monkeypatch):
    """App with the broadcaster stubbed out (covered by test_stream.py)."""
    config.save_config({"music_dir": str(music_dir)})
    library.scan_library(str(music_dir))

    from app import server
    from app.stream import BROADCASTER
    from app.scheduler import SCHEDULER

    monkeypatch.setattr(BROADCASTER, "start", lambda: None)
    monkeypatch.setattr(BROADCASTER, "stop", lambda: None)
    monkeypatch.setattr(SCHEDULER, "start", lambda: None)
    SCHEDULER.clear()
    with TestClient(server.app) as c:
        yield c


def test_status_endpoint(client):
    body = client.get("/api/status").json()
    assert body["library"]["total"] == 3
    assert "now_playing" in body and "config" in body


def test_health_endpoint(client):
    body = client.get("/api/health").json()
    assert "llm" in body and "tts" in body
    assert isinstance(body["llm"]["ok"], bool)


def test_index_page_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Virtual DJ" in resp.text


def test_static_assets_served(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_list_tracks_and_search(client):
    # "Band Three - Gamma" is untagged but has a clean filename guess, which is
    # kept playable (web confirmation no longer excludes a usable guess), so all
    # three files are playable.
    tracks = client.get("/api/library/tracks").json()
    assert len(tracks) == 3
    found = client.get("/api/library/tracks?search=Alpha").json()
    assert [t["title"] for t in found] == ["Alpha"]


def test_genres_endpoint(client):
    genres = {g["genre"] for g in client.get("/api/library/genres").json()}
    assert {"Rock", "Pop"} <= genres


def test_config_get_and_patch(client):
    assert client.get("/api/config").json()["dj"]["enabled"] is True
    updated = client.put("/api/config", json={"dj": {"talk_min": 1, "talk_max": 6}}).json()
    assert updated["dj"]["talk_min"] == 1
    assert updated["dj"]["talk_max"] == 6
    assert client.get("/api/config").json()["dj"]["talk_min"] == 1


def test_config_rejects_non_object(client):
    assert client.put("/api/config", json=["nope"]).status_code == 400


def test_queue_lifecycle(client):
    ids = [t["id"] for t in client.get("/api/library/tracks").json()]
    added = client.post("/api/queue", json={"track_ids": ids[:2]}).json()
    assert added["added"] == 2

    queue = client.get("/api/queue").json()
    assert len(queue) >= 2
    uid = queue[0]["uid"]

    assert client.post(f"/api/queue/{uid}/move", json={"index": 1}).status_code == 200
    assert client.delete(f"/api/queue/{uid}").status_code == 200
    assert client.delete(f"/api/queue/{uid}").status_code == 404
    assert client.post("/api/queue/clear").status_code == 200


def test_queue_replace(client):
    ids = [t["id"] for t in client.get("/api/library/tracks").json()]
    client.post("/api/queue", json={"track_ids": ids})
    body = client.post("/api/queue",
                       json={"track_ids": ids[:1], "replace": True}).json()
    assert body["added"] == 1
    assert len(body["queue"]) == 1


def test_move_unknown_uid_is_404(client):
    assert client.post("/api/queue/999999/move", json={"index": 0}).status_code == 404


def test_transport_controls(client):
    assert client.post("/api/transport/pause").json()["paused"] is True
    assert client.post("/api/transport/resume").json()["paused"] is False
    assert client.post("/api/transport/skip").json()["ok"] is True


def test_preset_endpoints(client):
    client.put("/api/config", json={"playback": {"genres": ["Rock"]}})
    assert client.post("/api/presets", json={"name": "rocknight"}).status_code == 200
    assert [p["name"] for p in client.get("/api/presets").json()] == ["rocknight"]

    client.put("/api/config", json={"playback": {"genres": ["Pop"]}})
    assert client.post("/api/presets/rocknight/apply").status_code == 200
    assert client.get("/api/config").json()["playback"]["genres"] == ["Rock"]

    assert client.delete("/api/presets/rocknight").status_code == 200
    assert client.post("/api/presets/rocknight/apply").status_code == 404


def test_preset_requires_name(client):
    assert client.post("/api/presets", json={"name": "  "}).status_code == 400


def test_scan_endpoint_triggers_scan(client, music_dir):
    body = client.post("/api/library/scan",
                       json={"music_dir": str(music_dir)}).json()
    assert "running" in body
    assert client.get("/api/library/scan").status_code == 200


def test_dj_voices_endpoint(client):
    body = client.get("/api/dj/voices").json()
    assert "voices" in body and "current" in body


def test_dj_audio_path_traversal_is_blocked(client):
    assert client.get("/api/dj/audio/..%2f..%2fconfig.json").status_code in (404, 400)
    assert client.get("/api/dj/audio/nope.mp3").status_code == 404


def test_history_endpoint(client):
    db.record_play(library.query_tracks()[0]["id"])
    rows = client.get("/api/history").json()
    assert len(rows) == 1
    assert rows[0]["title"]


def test_websocket_sends_state(client):
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "state"
        assert "listeners" in msg["data"]


def test_stream_endpoint_is_registered(client):
    """Header/behaviour checks for the infinite stream live in the E2E test
    (tests/test_e2e.py) — Starlette's TestClient cannot cleanly close an
    unbounded response, so it is exercised against a real uvicorn server."""
    routes = {getattr(r, "path", None) for r in client.app.routes}
    assert "/stream.mp3" in routes


def test_programs_endpoint_exposes_the_switchable_selection(client):
    """The Programs card's payload: the rotation, each theme's on/off state, and
    the coverage numbers behind its summary line."""
    # A program theme needs at least `size` tracks, so give the tiny scanned
    # library two real genres and lower the program size.
    conn = db.connect()
    for genre in ("AAA", "BBB"):
        for n in range(3):
            conn.execute(
                "INSERT INTO tracks(path,title,artist,album,genre,year,duration,"
                "mtime,size,missing,excluded,meta_source) VALUES(?,?,?,?,?,?,?,?,?,"
                "0,0,'tags')",
                (f"/m/{genre}/{n}.mp3", f"{genre} {n}", f"{genre} Band", "Al",
                 genre, "1999", 200.0, 1, 1),
            )
    conn.commit()
    client.put("/api/config", json={"playback": {"program": {"size": 2}}})

    body = client.get("/api/programs").json()
    assert body["strategy"] == "genre"
    assert body["limit"] == 20
    assert body["size"] == 2
    labels = [t["label"] for t in body["themes"]]
    assert "AAA" in labels and "BBB" in labels
    assert all(t["disabled"] is False for t in body["themes"])
    assert body["selected"] == len(body["themes"])
    assert body["candidate_tracks"] >= 6
    assert body["library_tracks"] >= 6

    # Switching one off through the same PUT the card uses must round-trip.
    label = "AAA"
    resp = client.put("/api/config", json={"playback": {"program": {
        "disabled": {"genre": [label], "artist": [], "decade": []}}}})
    assert resp.status_code in (200, 204)
    after = client.get("/api/programs").json()
    off = [t for t in after["themes"] if t["label"] == label]
    assert off and off[0]["disabled"] is True
    assert after["selected"] == len(after["themes"]) - 1
    assert after["disabled"]["genre"] == [label]


def test_config_write_refuses_an_unknown_key(client):
    """A typo'd payload key must fail loudly, not be stored and ignored."""
    r = client.put("/api/config", json={"playback": {"repeat_windows": 50}})
    assert r.status_code == 422
    assert "playback.repeat_windows" in str(r.json())
    # ...while a valid write still lands, and lands in the file the app reads.
    assert client.put("/api/config", json={"playback": {"repeat_window": 45}}).status_code == 200
    try:
        assert client.get("/api/config").json()["playback"]["repeat_window"] == 45
    finally:
        client.put("/api/config", json={"playback": {"repeat_window": 50}})


def test_cache_endpoints_report_and_sweep(client):
    """The Library card's temporary-data row and its "Clean now" button."""
    stats = client.get("/api/cache").json()
    assert {"dj_files", "dj_bytes", "enabled", "last_removed_bytes"} <= set(stats)
    assert stats["enabled"] is True
    swept = client.post("/api/cache/clean").json()
    assert "report" in swept and "stats" in swept
    assert isinstance(swept["report"]["removed_files"], int)


def test_artists_endpoint_feeds_the_filter_suggestions(client):
    body = client.get("/api/library/artists").json()
    assert body["artists"], "the fixture library has artists"
    assert {"artist", "n"} <= set(body["artists"][0])


def test_artist_filter_narrows_the_queue(client):
    """playback.artists is settable from the UI's artist field and is honoured."""
    artists = client.get("/api/library/artists").json()["artists"]
    name = artists[0]["artist"]
    client.put("/api/config", json={"playback": {"artists": [name]}})
    try:
        # Exactly what the card's Apply does: the filter takes effect on the NEXT
        # fill, so the already-queued (unfiltered) items are dropped first.
        client.post("/api/queue/clear")
        queue = client.get("/api/queue?limit=50").json()      # a list of queue items
        assert queue, "the queue should still fill from a single-artist filter"
        assert {item["track"]["artist"] for item in queue} == {name}, "artist filter leaked"
    finally:
        client.put("/api/config", json={"playback": {"artists": []}})


def test_programs_endpoint_exposes_the_artist_run_cap(client):
    """The card needs the cap to describe the rule it is enforcing."""
    body = client.get("/api/programs").json()
    assert body["max_consecutive_artist"] == 2
    # ...and whether that rule can hold for the selection at hand, so the summary
    # never promises a cap the queue builder is about to relax.
    assert body["cap_enforceable"] is True


def test_programs_endpoint_reports_an_unenforceable_cap(client, music_dir):
    """Pick one artist and the no-long-runs rule cannot hold — say so."""
    artist = "Band One"                       # one of the fixture's three artists
    client.put("/api/config", json={"playback": {"artists": [artist]}})
    try:
        body = client.get("/api/programs").json()
        assert body["cap_enforceable"] is False, (
            f"one artist selected ({artist!r}) cannot keep 2 songs in a row apart")
    finally:
        client.put("/api/config", json={"playback": {"artists": []}})
    assert client.get("/api/programs").json()["cap_enforceable"] is True
