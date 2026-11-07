"""Voice installation, delegated to the TTS service.

The app used to download Piper models into ``data/voices`` itself. It no longer
owns a voice format at all: a voice belongs to the running engine, so making one
usable is the service's job (``POST /v1/voices/{id}/install``).

This module is the app-side front door for that:

* ``ensure_default_voices()`` — first boot; makes the configured voice usable,
  best-effort (a TTS container that is still downloading must not crash boot);
* ``install_voices(names)`` — explicit, used by the CLI and the web UI;
* ``python -m app.voices`` — CLI.
"""

from __future__ import annotations

import logging
from typing import Iterable

from . import config, tts_client

log = logging.getLogger(__name__)


def known_voices() -> list[str]:
    """Every voice the service advertises (installed or not)."""
    return [v["id"] for v in tts_client.voices()]


def install_voice(voice: str) -> bool:
    """Make one voice usable. Returns True on success (never raises)."""
    try:
        tts_client.install_voice(voice)
        return True
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        log.error("could not install voice %s: %s", voice, exc)
        return False


def install_voices(names: Iterable[str]) -> list[str]:
    """Install several voices; return the ids that succeeded."""
    done = []
    for name in names:
        if install_voice(name):
            done.append(name)
    return done


def ensure_default_voices() -> list[str]:
    """Best-effort install of the voice the station needs at minimum."""
    if tts_client.capabilities() is None:
        # Not the same as "the engine offers nothing": the service is simply not
        # up yet (it may still be downloading its model). Nothing to do, and
        # nothing to alarm the user about — the DJ card installs on demand.
        log.info("TTS service not reachable yet; voice install deferred")
        return []
    known = known_voices()
    if not known:
        # A service that advertises nothing (an engine with no catalogue) cannot
        # install anything either — do not ask it to.
        log.warning("the TTS service advertises no voices; nothing to install")
        return []
    voice = tts_client.default_voice() or str(config.get("dj.voice") or "")
    if not voice or voice not in known:
        return []
    if voice in tts_client.installed_voice_ids():
        return []
    log.info("first-run voice setup: asking the TTS service for %s", voice)
    return install_voices([voice])


def main() -> None:
    """CLI: ``python -m app.voices [--all] [voice ...]``.

    With no args, installs the configured default voice. ``--all`` installs every
    voice the running engine declares. Extra positional args are explicit ids.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Install TTS voices via the TTS service")
    parser.add_argument("voices", nargs="*", help="explicit voice ids to install")
    parser.add_argument("--all", action="store_true",
                        help="install every voice the engine offers")
    parser.add_argument("--list", action="store_true",
                        help="show what the service offers and exit")
    args = parser.parse_args()

    if args.list or (not args.voices and not args.all):
        catalogue = tts_client.voices()
        if not catalogue:
            print(f"the TTS service at {tts_client.base_url()} offers nothing "
                  f"({tts_client.health().get('detail')})")
            return
        for v in catalogue:
            mark = "installed" if v.get("installed") else "not installed"
            langs = "/".join(v.get("languages") or [])
            print(f"  {v['id']:<10} {v.get('name', ''):<12} {langs:<8} {mark}")
        if not args.voices and not args.all:
            print(f"\nconfigured voice: {config.get('dj.voice')} "
                  f"language: {config.get('dj.language')}")
            return

    targets = list(args.voices)
    if args.all:
        targets = known_voices()
    if not targets:
        targets = [tts_client.default_voice() or str(config.get("dj.voice") or "")]
    targets = [t for t in targets if t]
    if not targets:
        print("nothing to install")
        return
    done = install_voices(targets)
    print(f"installed {len(done)} voice(s): {', '.join(done) or 'none'}")
    missing = [t for t in targets if t not in done]
    if missing:
        print(f"FAILED: {', '.join(missing)}")


if __name__ == "__main__":
    main()
