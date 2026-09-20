"""Public-web search and page fetch for specialist / Evaluator / Executor fact-checking."""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

logger = logging.getLogger("aiEmployees.research")

MAX_SEARCH_RESULTS = 8
MAX_FETCH_BYTES = 400_000
MAX_FETCH_CHARS = 12_000
FETCH_TIMEOUT = 15
SEARCH_TIMEOUT = 20
MAX_REDIRECTS = 5
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

_SKIP_TAGS = {"script", "style", "noscript", "svg", "iframe"}


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        if tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "blockquote"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = " ".join(data.split())
        if text:
            self.parts.append(text)


def _label(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if "reddit.com" in host:
        return "reddit — only use if later replies in the same thread confirm it worked"
    if host.endswith(".gov") or host.endswith(".mil"):
        return "official"
    if any(
        token in host
        for token in (
            "docs.",
            "developer.",
            "developers.",
            "learn.",
            "support.",
            "help.",
            "kb.",
            "oem.",
        )
    ):
        return "likely official docs"
    return "confirm this is the vendor's official site before relying on it"


def _assert_public_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http(s) URLs are allowed.")
    host = parsed.hostname
    if not host or host.lower() in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("That host is not allowed.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve {host}.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError("That host is not allowed.")


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in {"reddit.com", "www.reddit.com"}:
        netloc = parsed.netloc.replace(parsed.hostname, "old.reddit.com", 1)
        return parsed._replace(netloc=netloc).geturl()
    return url


class _GuardedRedirect(HTTPRedirectHandler):
    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _assert_public_http_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _request(url: str, data: bytes | None = None, timeout: int = FETCH_TIMEOUT):
    _assert_public_http_url(url)
    request = Request(
        url,
        data=data,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,text/plain"},
    )
    return build_opener(_GuardedRedirect).open(request, timeout=timeout)


def _html_text(raw: bytes) -> str:
    parser = _VisibleText()
    try:
        parser.feed(raw.decode("utf-8", errors="replace"))
        parser.close()
    except Exception:
        return raw.decode("utf-8", errors="replace")
    text = re.sub(r"\n{3,}", "\n\n", " ".join(parser.parts))
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _clip(text: str) -> str:
    if len(text) <= MAX_FETCH_CHARS:
        return text
    return text[:MAX_FETCH_CHARS] + "\n… [truncated]"


def _format_results(rows: list[tuple[str, str, str]]) -> str:
    if not rows:
        return "No results. Try a more specific query, or fetch a known official URL."
    lines = []
    for title, href, body in rows:
        lines.append(f"- {title}\n  {href}\n  [{_label(href)}] {body}")
    return "\n".join(lines)


def _search_ddgs(query: str) -> list[tuple[str, str, str]]:
    from ddgs import DDGS

    rows: list[tuple[str, str, str]] = []
    found = DDGS().text(query, max_results=MAX_SEARCH_RESULTS)
    for item in found or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        href = str(item.get("href") or item.get("url") or "").strip()
        body = str(item.get("body") or item.get("snippet") or "").strip()
        if href:
            rows.append((title or href, href, body))
    return rows


def _unwrap_ddg_href(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in (parsed.hostname or "") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return href


def _search_ddg_html(query: str) -> list[tuple[str, str, str]]:
    from urllib.parse import urlencode

    payload = urlencode({"q": query, "kl": "us-en"}).encode("utf-8")
    with _request(
        "https://html.duckduckgo.com/html/", data=payload, timeout=SEARCH_TIMEOUT
    ) as response:
        raw = response.read(MAX_FETCH_BYTES)
    html = raw.decode("utf-8", errors="replace")
    pattern = re.compile(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        re.I | re.S,
    )
    snippet_pattern = re.compile(
        r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>|'
        r'<td[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</td>',
        re.I | re.S,
    )
    snippets = [
        re.sub("<[^>]+>", "", (m.group(1) or m.group(2) or "")).strip()
        for m in snippet_pattern.finditer(html)
    ]
    rows: list[tuple[str, str, str]] = []
    for index, match in enumerate(pattern.finditer(html)):
        href = _unwrap_ddg_href(match.group(1))
        title = re.sub("<[^>]+>", "", match.group(2)).strip()
        body = snippets[index] if index < len(snippets) else ""
        if href.startswith("http"):
            rows.append((title or href, href, body))
        if len(rows) >= MAX_SEARCH_RESULTS:
            break
    return rows


def run_search(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return "Provide a search query."
    if len(query) > 300:
        query = query[:300]
    rows: list[tuple[str, str, str]] = []
    try:
        rows = _search_ddgs(query)
    except Exception as exc:
        logger.warning("ddgs search failed (%s).", exc)
    if not rows:
        try:
            rows = _search_ddg_html(query)
        except Exception as html_exc:
            logger.warning("DuckDuckGo HTML search failed (%s).", html_exc)
            if not rows:
                return (
                    f"Search failed ({type(html_exc).__name__}: {html_exc}). "
                    "Fetch a known official URL instead."
                )
    return _format_results(rows)


def run_fetch(url: str) -> str:
    url = _normalize_url((url or "").strip())
    if not url:
        return "Provide a URL to fetch."
    try:
        _assert_public_http_url(url)
    except ValueError as exc:
        return str(exc)
    try:
        with _request(url, timeout=FETCH_TIMEOUT) as response:
            content_type = (response.headers.get("Content-Type") or "").lower()
            raw = response.read(MAX_FETCH_BYTES + 1)
            final_url = response.geturl()
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return f"Could not fetch that page ({type(exc).__name__}: {exc})."
    truncated = len(raw) > MAX_FETCH_BYTES
    raw = raw[:MAX_FETCH_BYTES]
    if "html" in content_type or raw.lstrip()[:15].lower().startswith(
        (b"<!doctype html", b"<html")
    ):
        body = _html_text(raw)
    else:
        body = raw.decode("utf-8", errors="replace")
    note = f"Fetched {final_url} [{_label(final_url)}]"
    if truncated:
        note += " (truncated)"
    return note + "\n\n" + _clip(body)


def research_tools():
    from crewai.tools import tool

    @tool("search_web")
    def search_web(query: str) -> str:
        """Search the public web. Prefer official company, OEM, and vendor docs.
        For field reports, search site:reddit.com and only trust threads whose
        replies confirm the approach actually worked. query: search string."""
        return run_search(query)

    @tool("fetch_url")
    def fetch_url(url: str) -> str:
        """Fetch a public https page as readable text. Use for official docs,
        OEM pages, and Reddit threads you need to verify. url: full URL."""
        return run_fetch(url)

    return [search_web, fetch_url]
