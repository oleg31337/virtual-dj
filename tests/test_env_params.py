"""Environment-variable contract: every documented knob reaches the app.

Two invariants earn their place here:

1. An EMPTY value means "unset". Compose forwards ``"${VAR:-}"`` for optional
   knobs, so an undefined ``.env`` entry arrives as ``""`` — with a plain
   ``os.environ.get(name, default)`` that blanked real defaults (an unset
   ``VDJ_OLLAMA_URL`` left the DJ with an empty LLM endpoint) and
   ``int("")`` crashed the app at import time on an empty ``ICECAST_PORT=``.
2. Every variable compose forwards into the container is documented in
   ``.env.example``, so the example file cannot drift behind the stack.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from app import config

ROOT = Path(__file__).resolve().parent.parent


# --- the helpers -------------------------------------------------------------

def test_env_str_treats_empty_and_blank_as_unset(monkeypatch):
    monkeypatch.delenv("VDJ_TEST_KNOB", raising=False)
    assert config.env_str("VDJ_TEST_KNOB", "default") == "default"
    monkeypatch.setenv("VDJ_TEST_KNOB", "")
    assert config.env_str("VDJ_TEST_KNOB", "default") == "default"
    monkeypatch.setenv("VDJ_TEST_KNOB", "   ")
    assert config.env_str("VDJ_TEST_KNOB", "default") == "default"
    monkeypatch.setenv("VDJ_TEST_KNOB", "  http://x:1  ")
    assert config.env_str("VDJ_TEST_KNOB", "default") == "http://x:1"


def test_env_str_allows_an_empty_default(monkeypatch):
    """Some knobs are legitimately blank (icecast.public_host)."""
    monkeypatch.setenv("VDJ_TEST_KNOB", "")
    assert config.env_str("VDJ_TEST_KNOB", "") == ""


def test_env_int_never_raises(monkeypatch):
    for raw in (None, "", "  ", "junk", "1.5"):
        if raw is None:
            monkeypatch.delenv("VDJ_TEST_KNOB", raising=False)
        else:
            monkeypatch.setenv("VDJ_TEST_KNOB", raw)
        assert config.env_int("VDJ_TEST_KNOB", 8008) == 8008, raw
    monkeypatch.setenv("VDJ_TEST_KNOB", " 9001 ")
    assert config.env_int("VDJ_TEST_KNOB", 8008) == 9001
    monkeypatch.setenv("VDJ_TEST_KNOB", "-5")
    assert config.env_int("VDJ_TEST_KNOB", 0) == -5


def test_env_flag_reads_truthy_spellings(monkeypatch):
    for raw in ("1", "true", "TRUE", "yes", "on", " On "):
        monkeypatch.setenv("VDJ_TEST_KNOB", raw)
        assert config.env_flag("VDJ_TEST_KNOB") is True, raw
    for raw in ("0", "false", "no", "off", "junk"):
        monkeypatch.setenv("VDJ_TEST_KNOB", raw)
        assert config.env_flag("VDJ_TEST_KNOB") is False, raw
    monkeypatch.setenv("VDJ_TEST_KNOB", "")
    assert config.env_flag("VDJ_TEST_KNOB", default=True) is True
    monkeypatch.delenv("VDJ_TEST_KNOB", raising=False)
    assert config.env_flag("VDJ_TEST_KNOB", default=True) is True


# --- the effective defaults, as compose delivers them ------------------------

PROBE = """
import json, os, sys
sys.path.insert(0, os.environ["PROJ"])
from app import config
print(json.dumps({
    "llm.base_url": config.DEFAULTS["llm"]["base_url"],
    "llm.model": config.DEFAULTS["llm"]["model"],
    "icecast.mount": config.DEFAULTS["icecast"]["mount"],
    "icecast.source_password": config.DEFAULTS["icecast"]["source_password"],
    "icecast.port": config.DEFAULTS["icecast"]["port"],
    "icecast.public_port": config.DEFAULTS["icecast"]["public_port"],
    "icecast.public_host": config.DEFAULTS["icecast"]["public_host"],
    "icecast.hostname": config.DEFAULTS["icecast"]["hostname"],
    "icecast.enabled": config.DEFAULTS["icecast"]["enabled"],
    "music_dir": config.DEFAULTS["music_dir"],
    "loudness.workers": config.DEFAULTS["loudness"]["workers"],
}))
"""


def effective_defaults(extra_env: dict[str, str]) -> dict:
    """Import app.config in a fresh process with exactly ``extra_env`` set."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("VDJ_")}
    env.update({"PROJ": str(ROOT), "PATH": os.environ.get("PATH", "")})
    env.update(extra_env)
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


# what `docker compose up` forwards for a .env that leaves the optional knobs
# undefined: the literal empty string
COMPOSE_UNSET = {
    "VDJ_OLLAMA_URL": "", "VDJ_OLLAMA_MODEL": "",
    "VDJ_ICECAST_ENABLED": "1", "VDJ_ICECAST_HOST": "127.0.0.1",
    "VDJ_ICECAST_PORT": "8008", "VDJ_ICECAST_MOUNT": "virtualdj",
    "VDJ_ICECAST_SOURCE_PASSWORD": "hackme", "VDJ_ICECAST_PUBLIC_PORT": "8008",
    "VDJ_ICECAST_PUBLIC_HOST": "", "VDJ_ICECAST_HOSTNAME": "virtual-dj",
    "VDJ_ICECAST_ADMIN_PASSWORD": "", "VDJ_ICECAST_RELAY_PASSWORD": "",
    "VDJ_LOG_LEVEL": "info", "VDJ_LOUDNESS_WORKERS": "2",
    "VDJ_NO_VOICE_DOWNLOAD": "0", "VDJ_HOST": "0.0.0.0", "VDJ_PORT": "8420",
    "VDJ_DATA_DIR": "/data", "VDJ_MUSIC_DIR": "/music",
}


def test_compose_forwards_empty_strings_without_blanking_defaults():
    got = effective_defaults(COMPOSE_UNSET)
    assert got["llm.base_url"] == "http://127.0.0.1:11434", "empty URL must not win"
    assert got["llm.model"] == "qwen3.5:9b"
    assert got["icecast.mount"] == "virtualdj"
    assert got["icecast.source_password"] == "hackme"
    assert got["icecast.port"] == 8008
    assert got["icecast.public_port"] == 8008
    assert got["icecast.public_host"] == "", "blank is a valid value here"
    assert got["icecast.hostname"] == "virtual-dj"
    assert got["loudness.workers"] == 2
    assert got["music_dir"] == "/music"
    assert got["icecast.enabled"] is True


def test_env_values_win_when_set():
    got = effective_defaults({**COMPOSE_UNSET,
                              "VDJ_OLLAMA_URL": "http://192.168.1.222:11434",
                              "VDJ_OLLAMA_MODEL": "qwen3:8b",
                              "VDJ_LOUDNESS_WORKERS": "5",
                              "ICECAST_PORT_AS_VDJ": "1"})
    assert got["llm.base_url"] == "http://192.168.1.222:11434"
    assert got["llm.model"] == "qwen3:8b"
    assert got["loudness.workers"] == 5


@pytest.mark.parametrize("blank", ["VDJ_ICECAST_PORT", "VDJ_ICECAST_PUBLIC_PORT",
                                   "VDJ_LOUDNESS_WORKERS", "VDJ_PORT"])
def test_blank_numeric_env_does_not_crash_the_app(blank):
    """`ICECAST_PORT=` (blank line value) must not break the import."""
    got = effective_defaults({**COMPOSE_UNSET, blank: ""})
    assert got["icecast.port"] in (8008, 0) or blank != "VDJ_ICECAST_PORT"
    assert isinstance(got["loudness.workers"], int)


def test_blank_log_level_does_not_crash_logging():
    out = subprocess.run(
        [sys.executable, "-c", "import app; print('ok')"], cwd=ROOT,
        env={**{k: v for k, v in os.environ.items() if not k.startswith("VDJ_")},
             "VDJ_LOG_LEVEL": "", "PATH": os.environ.get("PATH", "")},
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-1500:]
    assert "ok" in out.stdout


# --- .env.example completeness ----------------------------------------------

def test_every_var_compose_forwards_is_documented_in_the_example():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    # names compose interpolates from .env (the operator-facing knobs)
    wanted = set(re.findall(r"\$\{([A-Z0-9_]+)", compose))
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"^#?\s*([A-Z0-9_]+)=", example, re.M))
    missing = sorted(wanted - documented)
    assert not missing, f"compose uses {missing} but .env.example never mentions them"


def test_every_documented_var_has_an_explanation():
    """Each active entry in .env.example is preceded by a comment.

    The point of the example file is that an operator can understand every knob
    without reading the code, so an unexplained `VAR=` line is a defect.
    """
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    unexplained = []
    for i, line in enumerate(lines):
        if not re.match(r"^[A-Z0-9_]+=", line):
            continue
        window = "\n".join(lines[max(0, i - 4):i])
        if "#" not in window:
            unexplained.append(line.split("=")[0])
    assert not unexplained, f"no explanation above: {unexplained}"
