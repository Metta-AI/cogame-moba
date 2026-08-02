"""Read/write Coworld artifact URIs: file:// and http(s)://.

Conventions mirror the Coworld runtime contract (bitworld runtime /
Cookbook "Raw Docker Shape"):

- ``file://`` URIs strip the scheme and keep the absolute path, so the
  platform's mount pattern ``file:///coworld/out/results.json`` resolves
  to ``/coworld/out/results.json``. Parent directories are created on
  write.
- Plain scheme-less strings are treated as local paths.
- ``http(s)://`` reads are GET, writes are PUT with a Content-Type
  header (signed URLs on the hosted platform). Non-2xx raises.
"""

from __future__ import annotations

from pathlib import Path

import aiohttp

_HTTP_SCHEMES = ("http://", "https://")


def local_path(uri: str) -> Path | None:
    """The local path for a file:// URI or plain path, else None."""
    if uri.startswith("file://"):
        return Path(uri[len("file://"):])
    if "://" not in uri:
        return Path(uri)
    return None


async def read_uri(uri: str) -> bytes:
    path = local_path(uri)
    if path is not None:
        return path.read_bytes()
    if uri.startswith(_HTTP_SCHEMES):
        async with aiohttp.ClientSession() as session:
            async with session.get(uri) as resp:
                if not 200 <= resp.status < 300:
                    raise IOError(
                        f"GET {uri} failed with status {resp.status}")
                return await resp.read()
    raise ValueError(f"unsupported URI scheme: {uri}")


async def write_uri(uri: str, data: bytes,
                    content_type: str = "application/octet-stream") -> None:
    path = local_path(uri)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    if uri.startswith(_HTTP_SCHEMES):
        async with aiohttp.ClientSession() as session:
            async with session.put(
                    uri, data=data,
                    headers={"Content-Type": content_type}) as resp:
                if not 200 <= resp.status < 300:
                    body = (await resp.text())[:200]
                    raise IOError(
                        f"PUT {uri} failed with status {resp.status}: {body}")
                return
    raise ValueError(f"unsupported URI scheme: {uri}")
