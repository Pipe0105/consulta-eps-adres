import time
import pandas as pd
from playwright.sync_api import sync_playwright

URL = "https://www.adres.gov.co/consulte-su-eps"  # si realmente consultas en otra URL final, ponla aquí
INPUT_XLSX = "input.xlsx"
OUTPUT_XLSX = "output_eps.xlsx"
DOC_COL = "NUMERO"

def parse_kv_table(page, table_id: str) -> dict:
    """
    Tabla tipo GridViewBasica: 2 columnas (label, value)
    """
    data = {}
    rows = page.locator(f"#{table_id} tr").all()
    for r in rows[1:]:  # saltar header
        cells = r.locator("td").all()
        if len(cells) >= 2:
            k = cells[0].inner_text().strip()
            v = cells[1].inner_text().strip()
            data[k] = v
    return data

def parse_aff_table(page, table_id: str) -> dict:
    """
    Tabla tipo GridViewAfiliacion: header + 1+ filas
    Tomamos la PRIMERA fila (la vigente) o concatenamos si hay varias.
    """
    headers = [h.inner_text().strip() for h in page.locator(f"#{table_id} th").all()]
    rows = page.locator(f"#{table_id} tr").all()

    # si no hay datos
    if len(rows) < 2:
        return {}

    # primera fila de datos (después del header)
    first = rows[1].locator("td").all()
    if len(first) != len(headers):
        return {}

    out = {}
    for h, c in zip(headers, first):
        out[h] = c.inner_text().strip()
    return out

def main():
    df = pd.read_excel(INPUT_XLSX, dtype=str)
    df[DOC_COL] = df[DOC_COL].astype(str).str.strip()

    # columnas salida
    df["EPS_TIPO_ID"] = ""
    df["EPS_NUMERO_ID"] = ""
    df["EPS_NOMBRES"] = ""
    df["EPS_APELLIDOS"] = ""
    df["EPS_DEPTO"] = ""
    df["EPS_MPIO"] = ""

    df["EPS_ESTADO"] = ""
    df["EPS_ENTIDAD"] = ""
    df["EPS_REGIMEN"] = ""
    df["EPS_FECHA_AFILIACION"] = ""
    df["EPS_FECHA_FIN"] = ""
    df["EPS_TIPO_AFILIADO"] = ""
    df["EPS_ERROR"] = ""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # pon True cuando ya esté estable
        page = browser.new_page()
        page.goto(URL, wait_until="domcontentloaded")

        for i, row in df.iterrows():
            doc = (row.get(DOC_COL) or "").strip()
            if not doc or doc.lower() == "nan":
                df.at[i, "EPS_ERROR"] = "Documento vacío"
                continue

            try:
                # (re)abrir por seguridad
                page.goto(URL, wait_until="domcontentloaded")

                # llenar documento
                page.fill("#txtNumDoc", doc)

                # click consultar (ejecuta setRecaptchaToken + postback)
                page.click("#btnConsultar")

                # esperar que aparezcan tablas de resultado
                page.wait_for_selector("#GridViewBasica", timeout=30000)

                # tabla básica
                basic = parse_kv_table(page, "GridViewBasica")

                df.at[i, "EPS_TIPO_ID"] = basic.get("TIPO DE IDENTIFICACIÓN", "")
                df.at[i, "EPS_NUMERO_ID"] = basic.get("NÚMERO DE IDENTIFICACION", "")
                df.at[i, "EPS_NOMBRES"] = basic.get("NOMBRES", "")
                df.at[i, "EPS_APELLIDOS"] = basic.get("APELLIDOS", "")
                df.at[i, "EPS_DEPTO"] = basic.get("DEPARTAMENTO", "")
                df.at[i, "EPS_MPIO"] = basic.get("MUNICIPIO", "")

                # tabla afiliación (puede tardar o no existir)
                if page.locator("#GridViewAfiliacion").count() > 0:
                    aff = parse_aff_table(page, "GridViewAfiliacion")
                    df.at[i, "EPS_ESTADO"] = aff.get("ESTADO", "")
                    df.at[i, "EPS_ENTIDAD"] = aff.get("ENTIDAD", "")
                    df.at[i, "EPS_REGIMEN"] = aff.get("REGIMEN", "")
                    df.at[i, "EPS_FECHA_AFILIACION"] = aff.get("FECHA DE AFILIACIÓN EFECTIVA", "")
                    df.at[i, "EPS_FECHA_FIN"] = aff.get("FECHA DE FINALIZACIÓN DE AFILIACIÓN", "")
                    df.at[i, "EPS_TIPO_AFILIADO"] = aff.get("TIPO DE AFILIADO", "")
                else:
                    df.at[i, "EPS_ERROR"] = "No apareció GridViewAfiliacion"

            except Exception as e:
                df.at[i, "EPS_ERROR"] = f"{type(e).__name__}: {e}"

            time.sleep(0.6)  # rate limit suave

        browser.close()

    df.to_excel(OUTPUT_XLSX, index=False)
    print("Listo:", OUTPUT_XLSX)

if __name__ == "__main__":
    main()
