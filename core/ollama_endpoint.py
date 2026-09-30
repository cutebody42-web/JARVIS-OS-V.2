"""Shared loopback-only Ollama endpoint normalization."""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse


def normalize_local_ollama_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Ollama URL must be a non-empty string")
    raw = value.strip()
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    if parsed.scheme != "http" or not parsed.netloc:
        raise ValueError("Local Ollama requires an HTTP loopback origin")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Ollama URL cannot contain credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Ollama URL must be an origin without path/query/fragment")

    host = parsed.hostname
    if host in {"0.0.0.0", "::"}:
        replacement = "127.0.0.1" if host == "0.0.0.0" else "::1"
        port = f":{parsed.port}" if parsed.port else ""
        netloc = f"[{replacement}]{port}" if ":" in replacement else replacement + port
        parsed = parsed._replace(netloc=netloc)
        host = replacement

    if host not in {"127.0.0.1", "::1"}:
        raise ValueError("Local Ollama endpoint must resolve to explicit loopback")

    return urlunparse(parsed).rstrip("/")
