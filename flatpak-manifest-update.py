#!/usr/bin/env python3
import asyncio
import hashlib
import logging
import sys
import urllib
from asyncio import BoundedSemaphore
from enum import Enum, unique
from string import Template
from typing import List, Tuple, Dict, Optional, Iterable, Pattern, Set
import re
from distutils.version import LooseVersion

import yaml
from aiohttp import ClientSession, ClientError

from flatpak_manifest_update.console import Foreground
from flatpak_manifest_update.logging import init_logging

try:
    from yaml import CLoader as YamlLoader
except ImportError:
    from yaml import Loader as YamlLoader


def load_yaml(stream):
    return yaml.load(stream, Loader=YamlLoader)


LOGGER = logging.getLogger("flatpak-manifest-updater")

TYPES = "archive", "file"

DEFAULT_HEADERS = {
    # https://github.com/aio-libs/aiohttp/issues/3904#issuecomment-632661245
    'Connection': 'keep-alive',
}


@unique
class State(Enum):
    UNKNOWN = "unknown"
    SKIPPED = "skipped"
    OK = "ok"
    NOT_FOUND = "not_found"
    WRONG_HASH = "wrong_hash"
    WRONG_SOURCE_URL = "wrong_source_url"
    WRONG_WATCH_URL = "wrong_watch_url"
    OUTDATED = "outdated"


ERROR_STATES = {
    State.NOT_FOUND,
    State.WRONG_HASH,
    State.WRONG_SOURCE_URL,
    State.WRONG_WATCH_URL,
    State.OUTDATED,
}

STATE_COLORS: Dict[State, Foreground] = {
    State.UNKNOWN: Foreground.MAGENTA_BOLD,
    State.SKIPPED: Foreground.BLUE_BOLD,
    State.OK: Foreground.GREEN_BOLD,
    State.NOT_FOUND: Foreground.RED_BOLD,
    State.WRONG_HASH: Foreground.RED_BOLD,
    State.WRONG_SOURCE_URL: Foreground.YELLOW_BOLD,
    State.WRONG_WATCH_URL: Foreground.YELLOW_BOLD,
    State.OUTDATED: Foreground.YELLOW_BOLD,
}


class Source:
    errors: List[str]

    def __init__(
            self,
            *,
            module: str,
            position: Tuple[int, int],
            type_: Optional[str] = None,
            current_url: Optional[str] = None,
            current_hash: Optional[str] = None,
            current_size: Optional[int] = None,
            real_hash: Optional[str] = None,
            current_version: Optional[str] = None,
            update_url: Optional[str] = None,
            version_pattern: Optional[str] = None,
            url_pattern: Optional[str] = None,
            errors: Optional[List[str]] = None,
            state: State = State.UNKNOWN,
            latest_version: Optional[str] = None,
            latest_url: Optional[str] = None,
            latest_hash: Optional[str] = None,
            latest_size: Optional[int] = None,
            ignored: bool = False,
    ):
        if version_pattern:
            version_pattern = Template(version_pattern).substitute(SUBSTITUTIONS)
        if url_pattern:
            url_pattern = Template(url_pattern).substitute(SUBSTITUTIONS)

        self.type = type_
        self.module = module
        self.position = position
        self.current_url = current_url
        self.current_hash = current_hash
        self.current_size = current_size
        self.real_hash = real_hash
        self.update_url = update_url
        self.version_pattern = version_pattern or url_pattern
        self.url_pattern = url_pattern
        self.state = state
        self.current_version = current_version
        self.errors = [] if errors is None else errors
        self.latest_version = latest_version
        self.latest_url = latest_url
        self.latest_hash = latest_hash
        self.latest_size = latest_size
        self.ignored = ignored

    @property
    def can_update(self) -> bool:
        return not self.missing_update_fields

    @property
    def missing_update_fields(self) -> List[str]:
        return [s for s in ("update_url", "url_pattern", "version_pattern") if not getattr(self, s)]

    def skip(self, reason: str) -> None:
        self.state = State.SKIPPED
        self.errors.append(reason)

    def add_error(self, error: str) -> None:
        self.errors.append(error)


async def main(argv: List[str]) -> int:
    init_logging()
    success = await process_manifest(argv[1], set(argv[2:]))
    return 0 if success else 1


async def process_manifest(path: str, only: Set[str] = None) -> bool:
    LOGGER.info("Manifest %s", path)

    with open(path) as f:
        manifest = load_yaml(f)

    modules = manifest['modules']
    sources = enqueue_sources(modules, only)
    queue = [s for s in sources if s.state != State.SKIPPED]
    errors = await check_sources(queue)
    for source, e in zip(queue, errors):
        if e is not None:
            source.add_error(f"{type(e).__name__}: {e}")
    print_summary(sources, unknown=True)

    return not any(not x.ignored and x.state in ERROR_STATES for x in sources)


VERSION_RE = r"(?P<version>\d+(?:\.\d+)+(?:[.+\-]\w+)*)"
VERSION_PURE_RE = r"(?P<version>\d+(?:\.\d+)+)"
VERSION_EVEN_RE = r"(?P<version>\d+\.\d*[02468](?:\.\d+)*(?:\.\w+)?)"

SUBSTITUTIONS = {
    "version": VERSION_RE,
    "pure_version": VERSION_PURE_RE,
    "even_version": VERSION_EVEN_RE,
}


def enqueue_sources(modules: dict, only: Set[str] = None) -> List[Source]:
    queue: List[Source] = []

    for module in modules:
        if not isinstance(module, dict):
            LOGGER.warning("Skipping module %r - not a dictionary.", module)
            continue

        try:
            module_name = module["name"]
        except KeyError:
            LOGGER.warning("Skipping module %r - 'name' is missing.", module)
            continue

        if only and module_name not in only:
            LOGGER.info("Skipping module %r as requested.", module_name)
            continue

        sources = module.get("sources", [])
        n_sources = len(sources)

        if not n_sources:
            LOGGER.warning("Skipping module %r - no sources.", module_name)
            continue

        for i, source in enumerate(sources, start=1):
            if not isinstance(source, dict):
                s = Source(module=module_name, position=(i, n_sources))
                s.skip("Not a dictionary.")
                queue.append(s)
                LOGGER.debug("Skipping module %r source #%d %r - not a dictionary.", module_name, i, source)
                continue

            params = source.get("x-update")
            if isinstance(params, dict):
                type_ = params.get("type", "html")
                project = params.get("project")

                if type_ == "pypi":
                    assert project
                    archive = params.get("archive", "tar.gz")
                    params.setdefault("url", f"https://pypi.org/project/{project}/")
                    params.setdefault("version-pattern", rf'{re.escape(project)}-$version\.{re.escape(archive)}')
                    params.setdefault(
                        "url-pattern",
                        r"https://files.pythonhosted.org/packages/[^/]+/[^/]+/[^/]+/" + params["version-pattern"])
                elif type_ == "gnome":
                    assert project
                    archive = params.get("archive", "tar.xz")
                    params.setdefault("url", f"https://download.gnome.org/sources/{project}/cache.json")
                    params.setdefault(
                        "url-pattern", rf'\d+\.\d+/{re.escape(project)}-$even_version\.{re.escape(archive)}')
                elif type_ == "github":
                    assert project
                    org, name = project.split("/")
                    releases = bool(params.get("releases", True))
                    archive = params.get("archive", "tar.xz" if releases else "tar.gz")

                    params.setdefault("url", f"https://github.com/{project}/releases/latest")

                    if releases:
                        params.setdefault(
                            "url-pattern",
                            rf"/{re.escape(project)}/releases/download/[^/]+/[-_\w]*[-v]$version\.{re.escape(archive)}"
                        )
                    else:
                        params.setdefault(
                            "url-pattern",
                            rf"/{re.escape(project)}/archive/refs/tags/[a-zA-Z]*$version\.{re.escape(archive)}"
                        )
                elif type_ == "ubuntu":
                    assert project
                    codename = params.get("codename", "groovy")
                    archive = params.get("archive", "tar.gz")
                    params.setdefault("url", f"https://launchpad.net/ubuntu/{codename}/+source/{project}")
                    params.setdefault(
                        "url-pattern",
                        rf"https://launchpad.net/ubuntu/\+archive/primary/\+sourcefiles/{re.escape(project)}/[^/]+/"
                        rf"{re.escape(project)}_$version\.orig\.{re.escape(archive)}"
                    )
                else:
                    assert type_ == "html"

                LOGGER.debug("Params of module %r source #%d: %r", module_name, i, params)

            s = Source(
                module=module_name,
                position=(i, n_sources),
                type_=source.get("type"),
                current_url=source.get("url"),
                current_hash=source.get("sha256"),
                update_url=params.get("url") if params else None,
                version_pattern=params.get("version-pattern") if params else None,
                url_pattern=params.get("url-pattern") if params else None,
                ignored=params.get("ignore", False) if params else False,
            )

            if s.type not in TYPES:
                LOGGER.debug("Skipping module %r source #%d - unsupported type %r", module_name, i, s.type)
                s.skip(f"Unsupported type {s.type!r}.")
                continue

            if not s.current_url and not s.can_update:
                LOGGER.warning("Skipping module %r source #%d - missing source 'url'.", module_name, i)
                s.skip("Missing source 'url'.")
                continue

            if not isinstance(params, dict):
                LOGGER.warning("Missing 'x-update' parameters in module %r source #%d.", module_name, i)

            queue.append(s)

    return queue


async def check_sources(sources: Iterable[Source], concurrency: int = 10) -> List[Optional[Exception]]:
    semaphore = BoundedSemaphore(concurrency)
    async with ClientSession() as session:
        return await asyncio.gather(*(check_source(semaphore, session, s) for s in sources), return_exceptions=True)


async def check_source(semaphore: BoundedSemaphore, session: ClientSession, source: Source) -> None:
    if current_url := source.current_url:
        try:
            async with semaphore:
                code, data = await fetch_data(session, current_url)
        except ClientError as e:
            source.state = State.NOT_FOUND
            source.add_error(f"Failed to download source URL {current_url!r} - {type(e).__name__}: {e}.")
        else:
            if code == 200:
                source.current_size = len(data)
                source.real_hash = await sha256(data)
                if source.current_hash != source.real_hash:
                    source.state = State.WRONG_HASH
                    source.add_error(f"The provided hash of {current_url!r} is wrong.")
            else:
                source.state = State.NOT_FOUND
                source.add_error(f"Failed to download source URL {current_url!r} - HTTP {code}.")

        url_pattern = re.compile(f"{source.url_pattern}$") if source.url_pattern else None
        if url_pattern:
            if (m := url_pattern.search(current_url)) is not None:
                source.current_version = m.group("version")
            else:
                source.add_error("Could not match 'url-pattern' in the current resource URL.")
        else:
            source.state = State.WRONG_WATCH_URL
            source.add_error("'url-pattern' is missing.")

    if missing := source.missing_update_fields:
        source.skip(f"Cannot update because {[s.replace('_', '-') for s in missing]} field(s) are missing.")
        return

    assert source.update_url and source.url_pattern and source.version_pattern
    update_url = source.update_url
    url_pattern = re.compile(source.url_pattern)
    version_pattern = re.compile(source.version_pattern)

    try:
        async with semaphore:
            code, html = await fetch_text(session, update_url)
    except ClientError as e:
        source.state = State.WRONG_WATCH_URL
        source.add_error(f"Failed to download update URL {update_url!r} - {type(e).__name__}: {e}.")
        return

    if code != 200:
        source.state = State.WRONG_WATCH_URL
        source.add_error(f"Failed to download update URL {update_url!r} - HTTP {code}.")
        return

    if not (versions := get_pattern(html, version_pattern)):
        source.state = State.WRONG_WATCH_URL
        source.add_error(f"No versions found on update URL {update_url!r}.")
        return

    if not (urls := get_pattern(html, url_pattern)):
        source.state = State.WRONG_WATCH_URL
        source.add_error(f"No resource URLs found on update URL {update_url!r}.")
        return

    latest_version_match = find_latest(versions)
    version = latest_version_match["version"]
    latest_url_match = find_latest(urls)
    latest_url = urllib.parse.urljoin(base=update_url, url=latest_url_match["match"])
    url_version = latest_url_match["version"]

    if version != url_version:
        source.state = State.WRONG_WATCH_URL
        source.add_error(
            f"Version from 'url-pattern' ({url_version!r}) differs from version from 'version-pattern' ({version!r})."
        )
        return

    source.latest_version = version
    source.latest_url = latest_url

    if source.current_url == latest_url:
        source.latest_hash = source.real_hash
        source.latest_size = source.current_size
        if source.state == State.UNKNOWN:
            source.state = State.OK
        return

    try:
        async with semaphore:
            code, data = await fetch_data(session, latest_url)
    except ClientError as e:
        source.state = State.NOT_FOUND
        source.add_error(f"Failed to download latest URL {latest_url!r} - {type(e).__name__}: {e}.")
        return

    if code != 200:
        source.state = State.NOT_FOUND
        source.add_error(f"Failed to download latest URL {latest_url!r} - HTTP {code}.")
        return

    source.latest_hash = await sha256(data)
    source.latest_size = len(data)
    source.state = State.OUTDATED


def print_summary(
        sources: List[Source], *, skipped: bool = False, unknown: bool = False, verbose: bool = False
) -> None:
    rows: List[tuple] = []
    for s in sources:
        if (not skipped and s.state == State.SKIPPED) or (not unknown and s.state == State.UNKNOWN):
            continue

        wrong_hash = s.current_hash != s.real_hash
        rows.append((STATE_COLORS[s.state].apply(
            f"{'IGNORED ' if s.ignored else ''}{s.state.name} {s.module} source {s.position[0]}/{s.position[1]}"),
        ))

        if s.errors:
            rows.append(("  Errors:",))
            rows += (("    " + e,) for e in s.errors)

        rows += [
            ("  Update URL:", s.update_url),
            ("  Current:",),
            ("    Version:", s.current_version),
            ("    Resource URL:", s.current_url),
            ("    Resource size:", s.current_size),
            ("    SHA256 - wrong:" if wrong_hash else "    SHA256:", s.current_hash),
            ("    SHA256 - correct:", s.real_hash) if wrong_hash else (),
        ]

        if verbose or s.latest_url != s.current_url:
            rows += [
                ("  Latest:",),
                ("    Version:", s.latest_version),
                ("    Resource URL:", s.latest_url),
                ("    Resource size:", s.latest_size),
                ("    SHA256:", s.latest_hash),
            ]

    if rows:
        info_len = max(len(r[0]) for r in rows if len(r) > 1)
        for r in rows:
            if len(r) > 1:
                print(f"{r[0]:{info_len}s} {r[1]}")
            elif r:
                print(r[0])


async def fetch_data(session: ClientSession, url: str, headers=None) -> Tuple[int, bytes]:
    if headers is None:
        headers = DEFAULT_HEADERS
    LOGGER.debug("Downloading: %s", url)
    async with session.get(url, headers=headers) as resp:
        return resp.status, await resp.read()


async def fetch_text(session: ClientSession, url: str, headers=None) -> Tuple[int, str]:
    if headers is None:
        headers = DEFAULT_HEADERS
    LOGGER.debug("Downloading: %s", url)
    async with session.get(url, headers=headers) as resp:
        return resp.status, await resp.text()


def get_pattern(html: str, pattern: Pattern[str]) -> List[Dict[str, str]]:
    return [{"match": m.group(0), **m.groupdict()} for m in pattern.finditer(html)]


def find_latest(items: List[Dict[str, str]], key: str = "version") -> Dict[str, str]:
    return max(items, key=lambda x: LooseVersion(x[key]))


async def sha256(data: bytes) -> str:
    hasher = hashlib.sha256()
    await asyncio.get_running_loop().run_in_executor(None, hasher.update, data)
    return hasher.hexdigest()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv)))
