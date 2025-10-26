"""
scraper/scrape.py

Scrape QuestDB docs under https://questdb.io/docs/
Produce JSONL file with entries:
{ "source_url": ..., "title": ..., "section": ..., "text": ... }
"""

import sys
from pathlib import Path
import argparse
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib.parse import urljoin, urlparse

# Add project root to sys.path so 'utils' can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

# utils should provide get_logger(name) and make_retry_session()
from utils import get_logger, make_retry_session

DOCS_ROOT = "https://questdb.io/docs/"
logger = get_logger("scraper")


@dataclass
class PageChunk:
    source_url: str
    title: str
    section: str
    text: str


def normalize_url(url: str) -> str:
    """Resolve, remove fragment, and strip trailing slash for consistent deduping."""
    try:
        parsed = urlparse(urljoin(DOCS_ROOT, url))
        normalized = parsed._replace(fragment="").geturl()
        if normalized.endswith("/") and len(normalized) > len(f"{parsed.scheme}://{parsed.netloc}/"):
            normalized = normalized.rstrip("/")
        return normalized
    except Exception:
        return url


def is_docs_url(url: str) -> bool:
    """Return True if URL points to questdb docs area."""
    try:
        p = urlparse(url)
        # Support both questdb.io and questdb.com after redirect
        return (p.netloc.endswith("questdb.io") or p.netloc.endswith("questdb.com")) and p.path.startswith("/docs")
    except Exception:
        return False


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def extract_main_container(soup: BeautifulSoup) -> Optional[Tag]:
    """Try common containers used in documentation sites; fallback to body."""
    # Prefer explicit, commonly-used docsite containers. Keep a forgiving fallback.
    selectors = [
        "main",
        "article",
        "div[role=main]",
        "#main-content",
        "#__docusaurus",
        ".theme-docs-content",
        ".theme-doc-markdown",
        ".markdown",
        "div[class*='markdown']",
        ".content",
        ".post",
        "div",
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            return el
    return soup.body


def section_chunking(container: Tag, max_chars: int = 1200, overlap: int = 200) -> List[Tuple[str, str]]:
    """Chunk sections by headers and sliding window if too long."""
    sections: List[Tuple[str, List[str]]] = []
    current_title = "Introduction"
    current_lines: List[str] = []

    def flush_section():
        nonlocal current_title, current_lines
        if current_lines:
            sections.append((current_title, current_lines))
            current_lines = []

    for node in container.descendants:
        if isinstance(node, Tag):
            name = node.name.lower()
            if name in ("h1", "h2", "h3"):
                header_text = clean_text(node.get_text(" "))
                if header_text:
                    flush_section()
                    current_title = header_text
            elif name in ("p", "li", "pre", "code", "div"):
                txt = clean_text(node.get_text("\n"))
                if txt:
                    if name in ("pre", "code") and len(txt) > 5000:
                        txt = txt[:5000] + "\n...[truncated]"
                    current_lines.append(txt)
        elif isinstance(node, NavigableString):
            continue

    flush_section()

    # Sliding-window chunking
    chunks: List[Tuple[str, str]] = []
    for sec_title, lines in sections:
        if not lines:
            continue
        section_text = "\n\n".join(lines).strip()
        if not section_text:
            continue
        if len(section_text) <= max_chars:
            chunks.append((sec_title, section_text))
            continue
        start = 0
        L = len(section_text)
        while start < L:
            end = min(L, start + max_chars)
            chunk = section_text[start:end].strip()
            if chunk:
                chunks.append((sec_title, chunk))
            if end == L:
                break
            start = max(end - overlap, start + 1)
    return chunks


def scrape_docs(base_url: str = DOCS_ROOT, max_pages: int = 1000, delay: float = 0.4, out_path: Optional[str] = None) -> List[PageChunk]:
    """Crawl docs and return list of PageChunk.

    If out_path is provided, write chunks to the JSONL file as they are discovered (streaming)
    and still return the full list.
    """
    session = make_retry_session()
    visited: Set[str] = set()
    q = deque([normalize_url(base_url)])
    queued: Set[str] = set(q)
    results: List[PageChunk] = []
    writer = None
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        writer = p.open("w", encoding="utf-8")

    while q and len(visited) < max_pages:
        url = q.popleft()
        queued.discard(url)
        # ensure source_url is always defined for the finally block
        source_url = url
        if url in visited:
            continue
        logger.info(f"Fetching: {url}")
        try:
            resp = session.get(url, timeout=20, headers={"User-Agent": "questdb-rag-scraper/1.0"})
            if resp.status_code != 200 or "html" not in resp.headers.get("content-type", ""):
                visited.add(url)
                time.sleep(delay)
                continue

            # Use final URL after redirects
            final_url = resp.url
            source_url = normalize_url(final_url)

            # Use the built-in parser for more predictable behavior across envs
            soup = BeautifulSoup(resp.text, "html.parser")

            # Queue same-site /docs/ links (avoid duplicates)
            for a in soup.find_all("a", href=True):
                try:
                    href = urljoin(source_url, a["href"])
                    normalized = normalize_url(href)
                    if is_docs_url(normalized) and normalized not in visited and normalized not in queued:
                        q.append(normalized)
                        queued.add(normalized)
                except Exception:
                    continue

            main = extract_main_container(soup)
            if not main:
                logger.debug(f"No main container found for {source_url}")
                visited.add(source_url)
                time.sleep(delay)
                continue

            h1 = main.find("h1")
            title_tag = h1 or soup.find("h1") or soup.find("title")
            title = clean_text(title_tag.get_text(" ")) if title_tag else source_url

            sec_chunks = section_chunking(main)
            if not sec_chunks:
                paras = [clean_text(p.get_text(" ")) for p in main.find_all("p") if clean_text(p.get_text(" "))]
                if paras:
                    text = "\n\n".join(paras)
                    sec_chunks = [("Content", text)]

            if sec_chunks:
                logger.info(f"Found {len(sec_chunks)} chunks on {source_url}")
            for sec_title, text in sec_chunks:
                if not text:
                    continue
                chunk = PageChunk(
                    source_url=source_url,
                    title=title,
                    section=sec_title,
                    text=text,
                )
                results.append(chunk)
                if writer:
                    rec = {
                        "source_url": chunk.source_url,
                        "title": chunk.title,
                        "section": chunk.section,
                        "text": chunk.text,
                    }
                    writer.write(json.dumps(rec, ensure_ascii=False) + "\n")

        except Exception as exc:
            logger.debug(f"Error while scraping {url}: {exc}")
        finally:
            # mark the URL visited (use the last-resolved source_url when available)
            try:
                visited.add(source_url)
            except Exception:
                visited.add(url)
            time.sleep(delay)

    if writer:
        writer.close()
    return results


def save_jsonl(chunks: Iterable[PageChunk], out_path: str) -> None:
    from pathlib import Path
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with p.open("w", encoding="utf-8") as f:
        for c in chunks:
            rec = {
                "source_url": c.source_url,
                "title": c.title,
                "section": c.section,
                "text": c.text
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    logger.info(f"Wrote {written} JSONL entries to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape QuestDB documentation into JSONL chunks.")
    parser.add_argument("--base-url", default=DOCS_ROOT)
    parser.add_argument("--out", default="data/questdb_docs.jsonl")
    parser.add_argument("--max-pages", type=int, default=1500)
    parser.add_argument("--delay", type=float, default=0.4)
    args = parser.parse_args()

    chunks = scrape_docs(args.base_url, max_pages=args.max_pages, delay=args.delay, out_path=args.out)
    print(f"Saved {len(chunks)} chunks to {args.out}")


if __name__ == "__main__":
    main()
