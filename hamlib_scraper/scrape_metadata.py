"""Download metadata CSVs from the HamLib chemistry directory tree."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from html.parser import HTMLParser
from http.client import IncompleteRead
import io
import json
from pathlib import Path, PurePosixPath
import sys
from tempfile import NamedTemporaryFile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

DEFAULT_URL = "https://portal.nersc.gov/cfs/m888/dcamps/hamlib/chemistry/"
MANIFEST = "manifest.json"
ACCEPTED = {"downloaded", "cached"}
MAX_BYTES = 64 * 1024 * 1024


def column_keys(headers):
    return {
        "".join(c for c in header.casefold() if c.isalnum()): header
        for header in headers if header is not None
    }


def find_counts(headers):
    keys = column_keys(headers)
    aliases = (
        ("nqubits", "numqubits", "numberofqubits", "qubits"),
        ("terms", "nterms", "numterms", "numberofterms"),
    )
    return tuple(
        next((keys[key] for key in names if key in keys), None)
        for names in aliases
    )


def is_metadata(data):
    with io.TextIOWrapper(io.BytesIO(data), encoding="utf-8-sig", newline="") as stream:
        return all(find_counts(next(csv.reader(stream), [])))


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temporary = Path(temp.name)
        try:
            temp.write(data)
            temp.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


class Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


class DirectoryTree:
    def __init__(self, url):
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.netloc or parts.username is not None:
            raise ValueError("Enter an HTTP(S) directory URL.")
        self.origin = (parts.scheme, parts.netloc)
        self.root_url = urlunsplit((*self.origin, parts.path.rstrip("/") + "/", "", ""))
        self.root_parts = self.path_parts(urlsplit(self.root_url).path)

    @staticmethod
    def path_parts(path):
        segments = tuple(unquote(part) for part in path.split("/") if part)
        if any(part in {".", ".."} or any(c in part for c in "/\\:\x00") for part in segments):
            raise ValueError(f"Invalid URL path: {path}")
        return segments

    def relative_path(self, url):
        parts = urlsplit(url)
        segments = self.path_parts(parts.path)
        if (
            (parts.scheme, parts.netloc) != self.origin
            or segments[:len(self.root_parts)] != self.root_parts
        ):
            raise ValueError(f"URL outside the requested directory: {url}")
        return PurePosixPath(*segments[len(self.root_parts):])

    def children(self, url, data):
        parser = Links()
        parser.feed(data.decode("utf-8", errors="replace"))
        directory = self.relative_path(url)
        for href in parser.hrefs:
            target = urlsplit(urljoin(url, href))
            if target.query or target.fragment:
                continue
            try:
                path = self.relative_path(target.geturl())
            except ValueError:
                continue
            if path == directory or path.parent != directory:
                continue
            is_directory = target.path.endswith("/")
            if is_directory or path.suffix.casefold() == ".csv":
                yield target.geturl(), path, is_directory


class ScopedRedirect(HTTPRedirectHandler):
    def __init__(self, tree, directory):
        self.tree = tree
        self.directory = directory

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        path = self.tree.relative_path(newurl)
        parts = urlsplit(newurl)
        if self.directory != parts.path.endswith("/") or parts.query:
            raise ValueError(f"Unexpected redirect: {newurl}")
        if not self.directory and path.suffix.casefold() != ".csv":
            raise ValueError(f"Expected a redirect to a CSV: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(tree, url, timeout):
    directory = urlsplit(url).path.endswith("/")
    opener = build_opener(ScopedRedirect(tree, directory))
    request = Request(url, headers={"User-Agent": "HamLibChemistryMetadata/1.0"})
    for attempt in range(3):
        try:
            with opener.open(request, timeout=timeout) as response:
                if directory and response.headers.get_content_type() not in {
                    "text/html", "application/xhtml+xml"
                }:
                    raise ValueError(f"Expected an HTML directory listing: {url}")
                data = response.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise ValueError(f"Response exceeds {MAX_BYTES // 1024**2} MiB")
                return data, response.geturl()
        except (URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
            permanent = isinstance(exc, HTTPError) and exc.code not in {
                408, 429, 500, 502, 503, 504
            }
            if attempt == 2 or permanent:
                raise
            time.sleep(2**attempt)


def scrape(root_url, out, *, refresh=False, timeout=25, workers=4):
    tree = DirectoryTree(root_url)
    manifest_path = out / MANIFEST
    manifest = {"root_url": tree.root_url, "directories": [], "files": [], "errors": []}
    cached_urls = {}
    if manifest_path.is_file() and not refresh:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["root_url"] == tree.root_url:
            cached_urls = {
                entry["path"]: entry["url"] for entry in previous["files"]
                if entry["status"] in ACCEPTED
            }

    def fetch_csv(item):
        relative, url = item
        destination = out.joinpath(*relative.parts)
        entry = {"path": relative.as_posix(), "url": url}
        try:
            if cached_urls.get(entry["path"]) == url and destination.is_file():
                try:
                    if is_metadata(destination.read_bytes()):
                        return {**entry, "status": "cached"}
                except (UnicodeError, csv.Error):
                    pass
            data, _ = fetch(tree, url, timeout)
            if not is_metadata(data):
                return {**entry, "status": "skipped", "reason": "Missing count headers"}
            atomic_write(destination, data)
            return {**entry, "status": "downloaded"}
        except (OSError, ValueError, csv.Error, IncompleteRead) as exc:
            return {**entry, "status": "error", "reason": str(exc)}

    pending = [tree.root_url]
    seen = {PurePosixPath(".")}
    candidates = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending:
            urls, pending = pending, []
            futures = [pool.submit(fetch, tree, url, timeout) for url in urls]
            for url, future in zip(urls, futures):
                try:
                    data, final_url = future.result()
                    manifest["directories"].append(tree.relative_path(url).as_posix())
                    for child, path, is_directory in tree.children(final_url, data):
                        if not is_directory:
                            candidates.setdefault(path, child)
                        elif path not in seen:
                            seen.add(path)
                            pending.append(child)
                except (OSError, ValueError, IncompleteRead) as exc:
                    manifest["errors"].append({"url": url, "reason": str(exc)})
        manifest["files"] = list(pool.map(fetch_csv, sorted(candidates.items())))

    manifest["errors"].extend(
        {"url": entry["url"], "reason": entry["reason"]}
        for entry in manifest["files"] if entry["status"] == "error"
    )
    atomic_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode())
    count = sum(entry["status"] in ACCEPTED for entry in manifest["files"])
    print(
        f"{len(manifest['directories'])} directories scanned, "
        f"{count} metadata CSVs, {len(manifest['errors'])} errors."
    )
    for error in manifest["errors"]:
        print(f"{error['url']}: {error['reason']}", file=sys.stderr)
    return 0 if count and not manifest["errors"] else 1


def add_scrape_options(parser):
    parser.add_argument("--root-url", default=DEFAULT_URL)
    parser.add_argument("--metadata-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout", type=float, default=25)
    parser.add_argument("--workers", type=int, default=4)


def scrape_from_args(args):
    if args.timeout <= 0 or args.workers < 1:
        raise ValueError("Timeout and worker count must be positive.")
    return scrape(
        args.root_url, args.metadata_dir, refresh=args.refresh,
        timeout=args.timeout, workers=args.workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_scrape_options(parser)
    try:
        raise SystemExit(scrape_from_args(parser.parse_args()))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
