import io
import anyio
import asyncio
import math
import os
import sys
import time
import uuid
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import Callable, Iterable, Tuple
import multiprocessing

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

from . import adres_lookup

START_URL = "https://www.adres.gov.co/consulte-su-eps"
RESULT_URL_PART = "BDUA_Internet/Pages/RespuestaConsulta.aspx"
DOC_COL_DEFAULT = "NUMERO"
NAME_COL_ALIASES = [
    "NOMBRE / RAZON SOCIAL",
    "NOMBRE/RAZON SOCIAL",
    "NOMBRE RAZON SOCIAL",
    "RAZON SOCIAL",
    "NOMBRE",
]


# --- templates (ruta absoluta, robusta) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app = FastAPI(title="Consulta masiva EPS (ADRES) - Sin API")
app.include_router(adres_lookup.router)
PROGRESS_LOCK = Lock()
PROGRESS_STATE = {}
PLAYWRIGHT_INSTALL_LOCK = Lock()
PLAYWRIGHT_INSTALL_ATTEMPTED = False
DEFAULT_WORKERS = max(1, min(4, (os.cpu_count() or 2)))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/progress/{job_id}")
def progress(job_id: str):
    with PROGRESS_LOCK:
        data = PROGRESS_STATE.get(job_id)
    if not data:
        return JSONResponse({"status": "unknown", "current": 0, "total": 0, "percent": 0}, status_code=404)
    return JSONResponse(data)

@app.post("/cancel/{job_id}")
def cancel(job_id: str):
    with PROGRESS_LOCK:
        data = PROGRESS_STATE.get(job_id)
        if not data:
            return JSONResponse({"status": "unknown"}, status_code=404)
        data["cancelled"] = True
        data["status"] = "canceled"
    return JSONResponse({"status": "canceled"})


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    doc_type: str = Form("CC"),
    headless: str = Form("1"),
    job_id: str | None = Form(None),
):
    raw = await file.read()
    df, doc_col, name_col = read_input_excel(raw)
    df[doc_col] = df[doc_col].astype(str).str.strip()
    if name_col:
        df[name_col] = df[name_col].astype(str).str.strip()

    out = df.copy()
    now = datetime.now().isoformat(timespec="seconds")
    out["TIPO_DOC"] = doc_type
    out["FECHA_PROCESO"] = now

    # Columnas salida
    out["EPS_TIPO_ID"] = ""
    out["EPS_NUMERO_ID"] = ""
    out["EPS_NOMBRES"] = ""
    out["EPS_APELLIDOS"] = ""
    out["EPS_DEPTO"] = ""
    out["EPS_MPIO"] = ""

    out["EPS_ESTADO"] = ""
    out["EPS_ENTIDAD"] = ""
    out["EPS_REGIMEN"] = ""
    out["EPS_FECHA_AFILIACION"] = ""
    out["EPS_FECHA_FIN"] = ""
    out["EPS_TIPO_AFILIADO"] = ""

    out["EPS_RESULT_URL"] = ""
    out["EPS_ERROR"] = ""

    use_headless = headless == "1"

    job_id = job_id or str(uuid.uuid4())
    init_progress(job_id, len(out))
    try:
        cancelled = await anyio.to_thread.run_sync(
            scrape_eps_parallel,
            out,
            doc_col,
            use_headless,
            job_id,
        )
        finish_progress(job_id, "canceled" if cancelled else "done")
    except Exception as exc:
        finish_progress(job_id, "error", error=str(exc))
        raise



    # Exportar excel
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="resultado")
    buf.seek(0)

    filename = f"resultado_eps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers
    )


def init_progress(job_id: str, total: int) -> None:
    with PROGRESS_LOCK:
        PROGRESS_STATE[job_id] = {
            "status": "running",
            "current": 0,
            "total": total,
            "percent": 0,
            "error": "",
            "cancelled": False,
        }


def update_progress(job_id: str, current: int, total: int) -> None:
    percent = int((current / total) * 100) if total else 0
    with PROGRESS_LOCK:
        if job_id in PROGRESS_STATE:
            PROGRESS_STATE[job_id].update(
                {
                    "current": current,
                    "total": total,
                    "percent": min(percent, 100),
                }
            )


def finish_progress(job_id: str, status: str, error: str = "") -> None:
    with PROGRESS_LOCK:
        if job_id in PROGRESS_STATE:
            PROGRESS_STATE[job_id].update(
                {
                    "status": status,
                    "percent": 100 if status == "done" else PROGRESS_STATE[job_id].get("percent", 0),
                    "error": error,
                }
            )
            
def is_cancelled(job_id: str) -> bool:
    with PROGRESS_LOCK:
        return PROGRESS_STATE.get(job_id, {}).get("cancelled", False)


def read_input_excel(raw_bytes: bytes) -> Tuple[pd.DataFrame, str, str | None]:
    df = pd.read_excel(io.BytesIO(raw_bytes), dtype=str)
    df.columns = [c.strip() for c in df.columns]

    if DOC_COL_DEFAULT in df.columns:
        return df, DOC_COL_DEFAULT, find_name_column(df)

    for c in ["DOCUMENTO", "CEDULA", "IDENTIFICACION", "ID"]:
        if c in df.columns:
            return df, c, find_name_column(df)

    return df, df.columns[0], find_name_column(df)


def find_name_column(df: pd.DataFrame) -> str | None:
    for col in NAME_COL_ALIASES:
        if col in df.columns:
            return col
    return None


def find_form_context(page):
    # El formulario puede estar en un iframe
    while True:
        if page.locator("#txtNumDoc").count() > 0:
            return page
        for fr in page.frames:
            if fr.locator("#txtNumDoc").count() > 0:
                return fr
        time.sleep(0.3)
        
def navigate_to_form(page, first_load: bool):
    if first_load:
        page.goto(START_URL, wait_until="domcontentloaded")
        return find_form_context(page)
    try:
        page.go_back(wait_until="domcontentloaded")
        return find_form_context(page)
    except Exception:
        page.goto(START_URL, wait_until="domcontentloaded")
        return find_form_context(page)




def parse_kv_table(ctx, table_id: str) -> dict:
    data = {}
    rows = ctx.locator(f"#{table_id} tr").all()
    if len(rows) < 2:
        return data
    for r in rows[1:]:
        tds = r.locator("td").all()
        if len(tds) >= 2:
            k = tds[0].inner_text().strip()
            v = tds[1].inner_text().strip()
            data[k] = v
    return data


def launch_browser(p, headless: bool):
    try:
        return p.chromium.launch(headless=headless)
    except PlaywrightError as exc:
        message = str(exc)
        if "Executable doesn't exist" in message:
            if ensure_playwright_browsers():
                return p.chromium.launch(headless=headless)
            raise RuntimeError(
                "Playwright no tiene instalado el navegador y no fue posible instalarlo. "
                "Ejecuta: playwright install"
            ) from exc
        raise
    
def ensure_playwright_browsers() -> bool:
    global PLAYWRIGHT_INSTALL_ATTEMPTED
    with PLAYWRIGHT_INSTALL_LOCK:
        if PLAYWRIGHT_INSTALL_ATTEMPTED:
            return False
        PLAYWRIGHT_INSTALL_ATTEMPTED = True

    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0





def scrape_eps_records(
    df: pd.DataFrame,
    doc_col: str,
    headless: bool,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> bool:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    with sync_playwright() as p:
        browser = launch_browser(p, headless=headless)
        context = browser.new_context()
        context.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in {"image", "media", "font", "stylesheet"}
            else route.continue_(),
        )
        page = context.new_page()
        page.set_default_timeout(0)
        page.set_default_navigation_timeout(0)
        
        total = len(df)
        first_load = True

        for i, row in df.iterrows():
            if cancel_check and cancel_check():
                browser.close()
                return True
            doc = str(row.get(doc_col, "")).strip()
            if not doc or doc.lower() == "nan":
                df.at[i, "EPS_ERROR"] = "Documento vacío"
                if progress_callback:
                    progress_callback(i + 1, total)
                continue

            try:
                ctx = navigate_to_form(page, first_load)
                first_load = False

                ctx.fill("#txtNumDoc", doc)
                ctx.click("#btnConsultar")

                try:
                    page.wait_for_url(f"**{RESULT_URL_PART}**")
                except PWTimeout:
                    pass

                df.at[i, "EPS_RESULT_URL"] = page.url

                # resultados normalmente en page
                result_ctx = page
                if page.locator("#GridViewBasica").count() == 0:
                    for fr in page.frames:
                        if fr.locator("#GridViewBasica").count() > 0:
                            result_ctx = fr
                            break

                result_ctx.wait_for_selector("#GridViewBasica")

                # parse GridViewBasica (2 cols)
                basic = parse_kv_table(result_ctx, "GridViewBasica")

                df.at[i, "EPS_TIPO_ID"] = basic.get("TIPO DE IDENTIFICACIÓN", "")
                df.at[i, "EPS_NUMERO_ID"] = basic.get("NÚMERO DE IDENTIFICACION", "")
                df.at[i, "EPS_NOMBRES"] = basic.get("NOMBRES", "")
                df.at[i, "EPS_APELLIDOS"] = basic.get("APELLIDOS", "")
                df.at[i, "EPS_DEPTO"] = basic.get("DEPARTAMENTO", "")
                df.at[i, "EPS_MPIO"] = basic.get("MUNICIPIO", "")

                # parse GridViewAfiliacion (primera fila)
                if result_ctx.locator("#GridViewAfiliacion").count() > 0:
                    headers = [h.inner_text().strip() for h in result_ctx.locator("#GridViewAfiliacion th").all()]
                    trs = result_ctx.locator("#GridViewAfiliacion tr").all()
                    if len(trs) >= 2:
                        first = trs[1].locator("td").all()
                        aff = {}
                        for h, c in zip(headers, first):
                            aff[h] = c.inner_text().strip()

                        df.at[i, "EPS_ESTADO"] = aff.get("ESTADO", "")
                        df.at[i, "EPS_ENTIDAD"] = aff.get("ENTIDAD", "")
                        df.at[i, "EPS_REGIMEN"] = aff.get("REGIMEN", "")
                        df.at[i, "EPS_FECHA_AFILIACION"] = aff.get("FECHA DE AFILIACIÓN EFECTIVA", "")
                        df.at[i, "EPS_FECHA_FIN"] = aff.get("FECHA DE FINALIZACIÓN DE AFILIACIÓN", "")
                        df.at[i, "EPS_TIPO_AFILIADO"] = aff.get("TIPO DE AFILIADO", "")

            except Exception as e:
                df.at[i, "EPS_ERROR"] = f"{type(e).__name__}: {e}"
            if progress_callback:
                progress_callback(i + 1, total)

        browser.close()
        return False


def scrape_eps_sync(df: pd.DataFrame, doc_col: str, headless: bool, job_id: str) -> bool:
    return scrape_eps_records(
        df,
        doc_col,
        headless,
        progress_callback=lambda current, total: update_progress(job_id, current, total),
        cancel_check=lambda: is_cancelled(job_id),
    )


def _scrape_eps_worker(records: list[dict], doc_col: str, headless: bool) -> list[dict]:
    df = pd.DataFrame.from_records(records).set_index("__index")
    scrape_eps_records(df, doc_col, headless)
    df.reset_index(inplace=True)
    return df.to_dict(orient="records")


def _chunk_records(df: pd.DataFrame, workers: int) -> Iterable[list[dict]]:
    total = len(df)
    if total == 0:
        return []
    chunk_size = max(1, math.ceil(total / workers))
    chunks = []
    for start in range(0, total, chunk_size):
        chunk = df.iloc[start : start + chunk_size].copy()
        chunk.reset_index(inplace=True)
        chunk.rename(columns={"index": "__index"}, inplace=True)
        chunks.append(chunk.to_dict(orient="records"))
    return chunks


def scrape_eps_parallel(df: pd.DataFrame, doc_col: str, headless: bool, job_id: str) -> bool:
    workers = int(os.getenv("EPS_WORKERS", str(DEFAULT_WORKERS)))
    if workers <= 1 or len(df) <= 1:
        return scrape_eps_sync(df, doc_col, headless, job_id)

    total = len(df)
    completed = 0

    chunks = _chunk_records(df, workers)
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
        future_map = {
            executor.submit(_scrape_eps_worker, chunk, doc_col, headless): chunk
            for chunk in chunks
        }
        for future in as_completed(future_map):
            if is_cancelled(job_id):
                for pending in future_map:
                    pending.cancel()
                return True
            results = future.result()
            for row in results:
                idx = row.pop("__index")
                for key, value in row.items():
                    df.at[idx, key] = value
                completed += 1
            update_progress(job_id, min(completed, total), total)

    return False