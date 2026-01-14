import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from fastapi import APIRouter
from pydantic import BaseModel, Field
from tenacity import AsyncRetrying, RetryError, retry_if_exception_type, stop_after_attempt, wait_exponential

START_URL = os.getenv("ADRES_START_URL", "https://www.adres.gov.co/consulte-su-eps")
DEFAULT_TIMEOUT = float(os.getenv("ADRES_TIMEOUT", "10"))
MAX_CONCURRENCY = int(os.getenv("ADRES_MAX_CONCURRENCY", "2"))
CACHE_TTL_SECONDS = int(os.getenv("ADRES_CACHE_TTL", "86400"))
CACHE_DB_PATH = os.getenv("ADRES_CACHE_DB", "./adres_cache.sqlite3")
BLOCK_THRESHOLD = int(os.getenv("ADRES_BLOCK_THRESHOLD", "3"))
BLOCK_PAUSE_SECONDS = int(os.getenv("ADRES_BLOCK_PAUSE_SECONDS", "300"))
USER_AGENT = os.getenv("ADRES_USER_AGENT", "adres-lookup/1.0")

BLOCK_KEYWORDS = (
    "captcha",
    "robot",
    "bloqueo",
    "acceso denegado",
    "demasiadas solicitudes",
    "too many requests",
)

NOT_FOUND_KEYWORDS = (
    "no se encontró",
    "no se encontraron",
    "no existen registros",
    "no existen resultados",
)

logger = logging.getLogger("adres_lookup")
logger.setLevel(logging.INFO)

router = APIRouter()


class LookupRequest(BaseModel):
    cedula: str = Field(..., min_length=3)
    tipo_doc: str | None = Field(default="CC")


class LookupResponse(BaseModel):
    status: str
    eps: str | None = None
    regimen: str | None = None
    detalle: str | None = None
    checked_at: str
    source: str


@dataclass
class CacheEntry:
    payload: dict
    expires_at: float
    checked_at: str


class SqliteCache:
    def __init__(self, path: str) -> None:
        self._path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=30)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eps_cache (
                    doc_hash TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    checked_at TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, doc_hash: str) -> CacheEntry | None:
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload, checked_at, expires_at FROM eps_cache WHERE doc_hash = ?",
                (doc_hash,),
            ).fetchone()
        if not row:
            return None
        payload, checked_at, expires_at = row
        if expires_at <= now:
            self.delete(doc_hash)
            return None
        return CacheEntry(payload=json.loads(payload), expires_at=expires_at, checked_at=checked_at)

    def set(self, doc_hash: str, payload: dict, checked_at: str, ttl_seconds: int) -> None:
        expires_at = time.time() + ttl_seconds
        with self._connect() as conn:
            conn.execute(
                "REPLACE INTO eps_cache (doc_hash, payload, checked_at, expires_at) VALUES (?, ?, ?, ?)",
                (doc_hash, json.dumps(payload), checked_at, expires_at),
            )
            conn.commit()

    def delete(self, doc_hash: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM eps_cache WHERE doc_hash = ?", (doc_hash,))
            conn.commit()


cache = SqliteCache(CACHE_DB_PATH)
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


class CircuitBreaker:
    def __init__(self) -> None:
        self.blocked_count = 0
        self.last_blocked_at: float | None = None

    def register_blocked(self) -> None:
        self.blocked_count += 1
        self.last_blocked_at = time.time()

    def reset(self) -> None:
        self.blocked_count = 0
        self.last_blocked_at = None

    def is_open(self) -> bool:
        if self.blocked_count < BLOCK_THRESHOLD:
            return False
        if self.last_blocked_at is None:
            return False
        return (time.time() - self.last_blocked_at) < BLOCK_PAUSE_SECONDS


circuit_breaker = CircuitBreaker()


def hash_doc(cedula: str) -> str:
    return hashlib.sha256(cedula.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_payload(soup: BeautifulSoup, cedula: str, tipo_doc: str | None) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    form = find_form(soup)
    if form is None:
        return payload

    for input_el in form.select("input"):
        name = input_el.get("name") or input_el.get("id")
        if not name:
            continue
        value = input_el.get("value", "")
        payload[name] = value

    if "txtNumDoc" in payload:
        payload["txtNumDoc"] = cedula
    else:
        payload["txtNumDoc"] = cedula

    if tipo_doc:
        if "ddlTipoDoc" in payload:
            payload["ddlTipoDoc"] = tipo_doc
        elif "tipo_doc" in payload:
            payload["tipo_doc"] = tipo_doc

    if "__EVENTTARGET" not in payload:
        payload["__EVENTTARGET"] = ""
    if "__EVENTARGUMENT" not in payload:
        payload["__EVENTARGUMENT"] = ""

    if "btnConsultar" in payload and not payload["btnConsultar"]:
        payload["btnConsultar"] = "Consultar"

    return payload


def find_form(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    for form in soup.select("form"):
        if form.select_one("#txtNumDoc") or form.select_one("input[name='txtNumDoc']"):
            return form
    return None


def extract_form_action(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    form = find_form(soup)
    if not form:
        return None
    action = form.get("action") or base_url
    return urljoin(base_url, action)


def parse_kv_table(soup: BeautifulSoup, table_id: str) -> Dict[str, str]:
    table = soup.find("table", {"id": table_id})
    if not table:
        return {}
    rows = table.find_all("tr")
    data: Dict[str, str] = {}
    for row in rows[1:]:
        cols = row.find_all("td")
        if len(cols) >= 2:
            key = cols[0].get_text(strip=True)
            value = cols[1].get_text(strip=True)
            data[key] = value
    return data


def parse_affiliation_table(soup: BeautifulSoup) -> Dict[str, str]:
    table = soup.find("table", {"id": "GridViewAfiliacion"})
    if not table:
        return {}
    rows = table.find_all("tr")
    if len(rows) < 2:
        return {}
    headers = [th.get_text(strip=True) for th in rows[0].find_all("th")]
    first_row = rows[1].find_all("td")
    data: Dict[str, str] = {}
    for header, cell in zip(headers, first_row):
        data[header] = cell.get_text(strip=True)
    return data


def parse_adres_response(html: str) -> Tuple[str, Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True).lower()
    if any(keyword in text for keyword in NOT_FOUND_KEYWORDS):
        return "not_found", {}

    basic = parse_kv_table(soup, "GridViewBasica")
    affiliation = parse_affiliation_table(soup)

    if not basic and not affiliation:
        return "not_found", {}

    data = {**basic, **affiliation}
    return "ok", data


def format_detalle(data: Dict[str, str]) -> str:
    parts = []
    for key in (
        "ESTADO",
        "TIPO DE AFILIADO",
        "FECHA DE AFILIACIÓN EFECTIVA",
        "FECHA DE FINALIZACIÓN DE AFILIACIÓN",
    ):
        value = data.get(key)
        if value:
            parts.append(f"{key.title()}: {value}")
    return "; ".join(parts)


def detect_blocked(status_code: int, body: str) -> bool:
    if status_code in {403, 429}:
        return True
    text = body.lower()
    return any(keyword in text for keyword in BLOCK_KEYWORDS)


async def fetch_with_retry(client: httpx.AsyncClient, method: str, url: str, **kwargs: Any) -> httpx.Response:
    retry = AsyncRetrying(
        retry=retry_if_exception_type(httpx.RequestError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    async for attempt in retry:
        with attempt:
            response = await client.request(method, url, **kwargs)
            if response.status_code >= 500:
                raise httpx.RequestError(f"Server error {response.status_code}", request=response.request)
            return response
    raise RetryError("Retries exhausted")

async def get_form_html(client: httpx.AsyncClient, start_url: str) -> Tuple[str, str]:
    response = await fetch_with_retry(client, "GET", start_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    iframe = soup.find("iframe")
    if iframe and iframe.get("src"):
        iframe_url = urljoin(start_url, iframe["src"])
        iframe_response = await fetch_with_retry(client, "GET", iframe_url)
        iframe_response.raise_for_status()
        return iframe_response.text, iframe_url
    return response.text, str(response.url)


async def lookup_live(cedula: str, tipo_doc: str | None, start_url: str | None = None) -> Tuple[str, Dict[str, str]]:
    start_url = start_url or START_URL
    headers = {"User-Agent": USER_AGENT}
    timeout = httpx.Timeout(DEFAULT_TIMEOUT)
    async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
        html, base_url = await get_form_html(client, start_url)
        soup = BeautifulSoup(html, "html.parser")
        action_url = extract_form_action(soup, base_url)
        if not action_url:
            raise ValueError("No se pudo encontrar el formulario de consulta.")

        payload = build_payload(soup, cedula, tipo_doc)
        response = await fetch_with_retry(client, "POST", action_url, data=payload)
        if detect_blocked(response.status_code, response.text):
            return "blocked", {}
        status, data = parse_adres_response(response.text)
        return status, data


@router.post("/eps-lookup", response_model=LookupResponse)
async def eps_lookup(request: LookupRequest) -> LookupResponse:
    if circuit_breaker.is_open():
        checked_at = now_iso()
        return LookupResponse(
            status="blocked",
            eps=None,
            regimen=None,
            detalle="Circuit breaker activo",
            checked_at=checked_at,
            source="live",
        )

    doc_hash = hash_doc(request.cedula)
    cached = cache.get(doc_hash)
    if cached:
        payload = cached.payload
        return LookupResponse(
            status=payload.get("status", "ok"),
            eps=payload.get("eps"),
            regimen=payload.get("regimen"),
            detalle=payload.get("detalle"),
            checked_at=cached.checked_at,
            source="cache",
        )

    start = time.perf_counter()
    checked_at = now_iso()
    status = "error"
    eps = None
    regimen = None
    detalle = None
    error_code = ""

    try:
        async with semaphore:
            status, data = await lookup_live(request.cedula, request.tipo_doc)
        if status == "blocked":
            circuit_breaker.register_blocked()
        else:
            circuit_breaker.reset()

        eps = data.get("ENTIDAD") or data.get("EPS")
        regimen = data.get("REGIMEN") or data.get("RÉGIMEN")
        detalle = format_detalle(data)

        payload = {
            "status": status,
            "eps": eps,
            "regimen": regimen,
            "detalle": detalle,
        }
        cache.set(doc_hash, payload, checked_at, CACHE_TTL_SECONDS)
    except (httpx.RequestError, ValueError, RetryError) as exc:
        error_code = type(exc).__name__
        status = "error"
        detalle = str(exc)
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "eps_lookup",
            extra={
                "doc_hash": doc_hash,
                "duration_ms": duration_ms,
                "status": status,
                "error_code": error_code,
                "checked_at": checked_at,
            },
        )

    return LookupResponse(
        status=status,
        eps=eps,
        regimen=regimen,
        detalle=detalle,
        checked_at=checked_at,
        source="live",
    )