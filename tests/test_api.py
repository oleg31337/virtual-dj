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


def test_artists_endpoint(client):
    artists = {a["artist"] for a in client.get("/api/library/artists").json()}
    assert "Band One" in artists


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
