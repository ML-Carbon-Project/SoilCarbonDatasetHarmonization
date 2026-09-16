from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USER_AGENT = "carbono-solo-collector/0.1"


def write_csv(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
    *,
    encoding: str = "utf-8",
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter=";",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def fetch_json(url: str, timeout: int = 30) -> object:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    return json.loads(_fetch_text(request, timeout))


def fetch_text(url: str, timeout: int = 60) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
        },
    )
    return _fetch_text(request, timeout)


def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
        },
    )
    return _fetch_bytes(request, timeout)


def post_text(
    url: str,
    data: Mapping[str, object] | str,
    timeout: int = 60,
) -> str:
    body = data if isinstance(data, str) else urlencode(data)
    request = Request(
        url,
        data=body.encode("utf-8"),
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        },
    )
    return _fetch_text(request, timeout)


def _fetch_text(request: Request, timeout: int) -> str:
    payload, charset = _read_response(request, timeout)
    try:
        return payload.decode(charset or "utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Error decoding {request.full_url}: {exc}") from exc


def _fetch_bytes(request: Request, timeout: int) -> bytes:
    payload, _charset = _read_response(request, timeout)
    return payload


def _read_response(request: Request, timeout: int) -> tuple[bytes, str | None]:
    try:
        with urlopen(request, timeout=timeout) as response:
            headers = getattr(response, "headers", None)
            get_content_charset = getattr(headers, "get_content_charset", None)
            charset = get_content_charset() if get_content_charset else None
            return response.read(), charset
    except HTTPError as exc:
        reason = getattr(exc, "reason", None) or getattr(exc, "msg", None)
        detail = f"{exc.code} {reason}" if reason else str(exc.code)
        raise RuntimeError(f"HTTP error fetching {request.full_url}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Error fetching {request.full_url}: {exc}") from exc
