import time
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

START_URL = "https://www.adres.gov.co/consulte-su-eps"
RESULT_URL_PART = "BDUA_Internet/Pages/RespuestaConsulta.aspx"

INPUT_XLSX = "input.xlsx"          # tu archivo
OUTPUT_XLSX = "output_eps.xlsx"    # salida
DOC_COL = "NUMERO"                 # columna del documento en tu Excel

async def parse_kv_table(ctx, table_id: str) -> dict:
    data = {}
    rows = await ctx.locator(f"#{table_id} tr").all()
    if len(rows) < 2:
        return data
    for r in rows[1:]:
        tds = await r.locator("td").all()
        if len(tds) >= 2:
            k = (await tds[0].inner_text()).strip()
            v = (await tds[1].inner_text()).strip()
            data[k] = v
    return data


def parse_aff_table_first_row(frame_or_page, table_id: str) -> dict:
    """GridViewAfiliacion: toma la primera fila de datos."""
    out = {}
    headers = [h.inner_text().strip() for h in frame_or_page.locator(f"#{table_id} th").all()]
    rows = frame_or_page.locator(f"#{table_id} tr").all()
    if len(rows) < 2 or not headers:
        return out

    first = rows[1].locator("td").all()
    if len(first) != len(headers):
        return out

    for h, c in zip(headers, first):
        out[h] = c.inner_text().strip()
    return out

async def find_form_context(page):
    if await page.locator("#txtNumDoc").count() > 0:
        return page
    for fr in page.frames:
        if await fr.locator("#txtNumDoc").count() > 0:
            return fr
    raise RuntimeError("No encontré #txtNumDoc (cambió la página o no cargó el iframe).")

def main():
    df = pd.read_excel(INPUT_XLSX, dtype=str)
    df[DOC_COL] = df[DOC_COL].astype(str).str.strip()

    # Columnas salida (básica)
    df["EPS_TIPO_ID"] = ""
    df["EPS_NUMERO_ID"] = ""
    df["EPS_NOMBRES"] = ""
    df["EPS_APELLIDOS"] = ""
    df["EPS_DEPTO"] = ""
    df["EPS_MPIO"] = ""

    # Afiliación
    df["EPS_ESTADO"] = ""
    df["EPS_ENTIDAD"] = ""
    df["EPS_REGIMEN"] = ""
    df["EPS_FECHA_AFILIACION"] = ""
    df["EPS_FECHA_FIN"] = ""
    df["EPS_TIPO_AFILIADO"] = ""

    # Debug/errores
    df["EPS_RESULT_URL"] = ""
    df["EPS_ERROR"] = ""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # cambia a True cuando esté estable
        page = browser.new_page()
        page.set_default_timeout(0)
        page.set_default_navigation_timeout(0)
        page.goto(START_URL, wait_until="domcontentloaded")

        for i, row in df.iterrows():
            doc = (row.get(DOC_COL) or "").strip()
            if not doc or doc.lower() == "nan":
                df.at[i, "EPS_ERROR"] = "Documento vacío"
                continue

            try:
                # Reabrir inicio para estado limpio
                page.goto(START_URL, wait_until="domcontentloaded")

                # Contexto (iframe o página)
                ctx = find_form_context(page)

                # Llenar y consultar
                ctx.fill("#txtNumDoc", doc)
                ctx.click("#btnConsultar")

                # Esperar a que navegue al resultado con tokenId
                try:
                    page.wait_for_url(f"**{RESULT_URL_PART}**")
                except PWTimeout:
                    # A veces la navegación ocurre dentro del iframe; intentamos detectarlo
                    # pero normalmente tokenId termina en la URL principal.
                    pass

                # Si el resultado está en otra pestaña/redirect, acá lo tomamos:
                current_url = page.url
                if RESULT_URL_PART not in current_url:
                    # Puede seguir en frame, entonces buscamos un frame con la tabla
                    # y si la encontramos, parseamos desde ahí sin URL.
                    pass

                df.at[i, "EPS_RESULT_URL"] = current_url

                # Contexto de resultado: normalmente es page (porque ya redirigió)
                result_ctx = page
                if page.locator("#GridViewBasica").count() == 0:
                    # buscar en frames por si el resultado quedó embebido
                    for fr in page.frames:
                        if fr.locator("#GridViewBasica").count() > 0:
                            result_ctx = fr
                            break

                # Esperar tablas
                result_ctx.wait_for_selector("#GridViewBasica")

                # Parse básico
                basic = parse_kv_table(result_ctx, "GridViewBasica")
                df.at[i, "EPS_TIPO_ID"] = basic.get("TIPO DE IDENTIFICACIÓN", "")
                df.at[i, "EPS_NUMERO_ID"] = basic.get("NÚMERO DE IDENTIFICACION", "")
                df.at[i, "EPS_NOMBRES"] = basic.get("NOMBRES", "")
                df.at[i, "EPS_APELLIDOS"] = basic.get("APELLIDOS", "")
                df.at[i, "EPS_DEPTO"] = basic.get("DEPARTAMENTO", "")
                df.at[i, "EPS_MPIO"] = basic.get("MUNICIPIO", "")

                # Parse afiliación (si existe)
                if result_ctx.locator("#GridViewAfiliacion").count() > 0:
                    aff = parse_aff_table_first_row(result_ctx, "GridViewAfiliacion")
                    df.at[i, "EPS_ESTADO"] = aff.get("ESTADO", "")
                    df.at[i, "EPS_ENTIDAD"] = aff.get("ENTIDAD", "")
                    df.at[i, "EPS_REGIMEN"] = aff.get("REGIMEN", "")
                    df.at[i, "EPS_FECHA_AFILIACION"] = aff.get("FECHA DE AFILIACIÓN EFECTIVA", "")
                    df.at[i, "EPS_FECHA_FIN"] = aff.get("FECHA DE FINALIZACIÓN DE AFILIACIÓN", "")
                    df.at[i, "EPS_TIPO_AFILIADO"] = aff.get("TIPO DE AFILIADO", "")
                else:
                    # no siempre hay afiliación
                    pass

            except Exception as e:
                df.at[i, "EPS_ERROR"] = f"{type(e).__name__}: {e}"

            # rate limit suave para no martillar el sitio
            time.sleep(0.7)

        browser.close()

    df.to_excel(OUTPUT_XLSX, index=False)
    print("Listo:", OUTPUT_XLSX)

if __name__ == "__main__":
    main()
