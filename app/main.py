import io
import anyio

import os
import time
from datetime import datetime
from typing import Tuple

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout



START_URL = "https://www.adres.gov.co/consulte-su-eps"
RESULT_URL_PART = "BDUA_Internet/Pages/RespuestaConsulta.aspx"
DOC_COL_DEFAULT = "NUMERO"

# --- templates (ruta absoluta, robusta) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app = FastAPI(title="Consulta masiva EPS (ADRES) - Sin API")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    doc_type: str = Form("CC"),
    headless: str = Form("0"),
):
    raw = await file.read()
    df, doc_col = read_input_excel(raw)
    df[doc_col] = df[doc_col].astype(str).str.strip()

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

    use_headless = (headless == "1")

    await anyio.to_thread.run_sync(scrape_eps_sync, out, doc_col, use_headless)



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


def read_input_excel(raw_bytes: bytes) -> Tuple[pd.DataFrame, str]:
    df = pd.read_excel(io.BytesIO(raw_bytes), dtype=str)
    df.columns = [c.strip() for c in df.columns]

    if DOC_COL_DEFAULT in df.columns:
        return df, DOC_COL_DEFAULT

    for c in ["DOCUMENTO", "CEDULA", "IDENTIFICACION", "ID"]:
        if c in df.columns:
            return df, c

    return df, df.columns[0]


def find_form_context(page):
    # El formulario puede estar en un iframe
    if page.locator("#txtNumDoc").count() > 0:
        return page
    for fr in page.frames:
        if fr.locator("#txtNumDoc").count() > 0:
            return fr
    raise RuntimeError("No encontré #txtNumDoc (cambió la página o no cargó el iframe).")


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


async def parse_aff_first_row(ctx, table_id: str) -> dict:
    out = {}
    headers_nodes = await ctx.locator(f"#{table_id} th").all()
    headers = [(await h.inner_text()).strip() for h in headers_nodes]

    rows = await ctx.locator(f"#{table_id} tr").all()
    if len(rows) < 2 or not headers:
        return out

    first = await rows[1].locator("td").all()
    if len(first) != len(headers):
        return out

    for h, c in zip(headers, first):
        out[h] = (await c.inner_text()).strip()
    return out



def scrape_eps_sync(df: pd.DataFrame, doc_col: str, headless: bool):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()

        for i, row in df.iterrows():
            doc = str(row.get(doc_col, "")).strip()
            if not doc or doc.lower() == "nan":
                df.at[i, "EPS_ERROR"] = "Documento vacío"
                continue

            try:
                page.goto(START_URL, wait_until="domcontentloaded")

                # formulario puede estar en iframe
                ctx = page
                if page.locator("#txtNumDoc").count() == 0:
                    found = False
                    for fr in page.frames:
                        if fr.locator("#txtNumDoc").count() > 0:
                            ctx = fr
                            found = True
                            break
                    if not found:
                        raise RuntimeError("No encontré #txtNumDoc en página ni en iframes")

                ctx.fill("#txtNumDoc", doc)
                ctx.click("#btnConsultar")

                try:
                    page.wait_for_url(f"**{RESULT_URL_PART}**", timeout=30000)
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

                result_ctx.wait_for_selector("#GridViewBasica", timeout=30000)

                # parse GridViewBasica (2 cols)
                basic = {}
                rows = result_ctx.locator("#GridViewBasica tr").all()
                for r in rows[1:]:
                    tds = r.locator("td").all()
                    if len(tds) >= 2:
                        k = tds[0].inner_text().strip()
                        v = tds[1].inner_text().strip()
                        basic[k] = v

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

            time.sleep(0.7)

        browser.close()
