#!/usr/bin/env python3
"""Lleva el marcador: IA local contra pronóstico oficial contra la realidad.

Dos órdenes:

    registrar   guarda lo que predijo el modelo local y lo que predice el IFS
                operativo de ECMWF, para las mismas horas exactas
    verificar   rellena, en las horas que ya pasaron, lo que de verdad ocurrió

Las predicciones se escriben antes del hecho y no se tocan nunca más. Solo se
añade la columna de la realidad. Con los meses, el registro dice quién acierta
sin que nadie tenga que creer en la palabra de nadie.

El registro es un JSONL: una línea por corrida.

    python marcador.py registrar --pronostico pronostico.json --registro clima.jsonl
    python marcador.py verificar --registro clima.jsonl
"""

import argparse
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

UA = {"User-Agent": "marcador-clima/1.0 (https://github.com/EfrainGaray/marcador-clima)"}


def traer(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        j = json.loads(r.read())
    h = j["hourly"]
    return {t: v for t, v in zip(h["time"], h["temperature_2m"]) if v is not None}


def clave(fecha_str):
    """'2026-08-16 06:00:00' -> '2026-08-16T06:00', como indexa Open-Meteo."""
    return fecha_str.replace(" ", "T")[:16]


def cmd_registrar(args):
    pron = json.loads(Path(args.pronostico).read_text())
    registro = Path(args.registro)
    analisis = pron["analisis_inicial_utc"]

    if registro.exists():
        for linea in registro.read_text().splitlines():
            if linea.strip() and json.loads(linea).get("analisis_utc") == analisis:
                print(f"ya hay registro del análisis {analisis}; nada que hacer")
                return

    lat, lon = pron["coords_pedidas"]
    horas = [clave(p["fecha_utc"]) for p in pron["pasos"]]

    try:
        oficial = traer(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                        f"&hourly=temperature_2m&models={args.modelo_oficial}"
                        f"&forecast_days=4&timezone=UTC"
                        # Sin esto Open-Meteo elige una celda terrestre de elevación
                        # parecida y ajusta por altura con un DEM de 90 m: contra el
                        # gridpoint crudo de AIFS eso metía 0.60 °C de sesgo, dos
                        # tercios del error que se quiere medir.
                        f"&cell_selection=nearest&elevation=nan")
    except Exception as e:
        print(f"aviso: no se pudo traer el oficial ({type(e).__name__}); queda en null")
        oficial = {}

    fila = {
        "analisis_utc": analisis,
        "registrado_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lugar": args.lugar,
        "coords_grilla": pron["coords_grilla"],
        "hardware": pron["hardware"],
        "segundos_pronostico": pron["segundos_pronostico"],
        "vram_pico_GB": pron.get("vram_pico_GB"),
        "pasos": [{"hora_utc": h,
                   "ia_local_C": p.get("t2m_C"),
                   "oficial_C": oficial.get(h),
                   "real_C": None,
                   "fuente_real": None}
                  for h, p in zip(horas, pron["pasos"])],
    }

    registro.parent.mkdir(parents=True, exist_ok=True)
    with registro.open("a") as fh:
        fh.write(json.dumps(fila, ensure_ascii=False) + "\n")
    print(f"registrado {analisis}: {len(fila['pasos'])} pasos")


def cmd_verificar(args):
    registro = Path(args.registro)
    if not registro.exists():
        print("no hay registro todavía")
        return

    filas = [json.loads(l) for l in registro.read_text().splitlines() if l.strip()]
    ahora = datetime.now(timezone.utc)

    pend = [p for f in filas for p in f["pasos"]
            if p["fuente_real"] != "era5"
            and datetime.fromisoformat(p["hora_utc"]).replace(tzinfo=timezone.utc) < ahora]
    if not pend:
        print("nada que verificar")
        return

    lat, lon = args.lat, args.lon
    horas = sorted({p["hora_utc"] for p in pend})
    desde = horas[0][:10]
    hasta = min(ahora.date(),
                datetime.fromisoformat(horas[-1]).date() + timedelta(days=1)).isoformat()

    definitivo, provisional = {}, {}
    try:
        # ERA5: la referencia, pero llega con un par de días de retraso
        definitivo = traer(f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}"
                           f"&longitude={lon}&start_date={desde}&end_date={hasta}"
                           f"&hourly=temperature_2m&timezone=UTC")
    except Exception as e:
        print(f"aviso: ERA5 no disponible ({type(e).__name__})")
    try:
        provisional = traer(f"https://api.open-meteo.com/v1/forecast?latitude={lat}"
                            f"&longitude={lon}&hourly=temperature_2m&past_days=7"
                            f"&forecast_days=1&timezone=UTC")
    except Exception as e:
        print(f"aviso: análisis reciente no disponible ({type(e).__name__})")

    n_def = n_prov = 0
    for f in filas:
        for p in f["pasos"]:
            if p["fuente_real"] == "era5":
                continue
            # El filtro de arriba decide QUE verificar, pero escribir sin
            # repetir la comprobacion marcaba como "real" horas que aun no
            # ocurren: las APIs devuelven el dia en curso completo, con las
            # horas futuras rellenadas por su propio modelo.
            if datetime.fromisoformat(p["hora_utc"]).replace(tzinfo=timezone.utc) >= ahora:
                continue
            h = p["hora_utc"]
            if h in definitivo:
                p["real_C"], p["fuente_real"] = round(definitivo[h], 2), "era5"
                n_def += 1
            elif p["real_C"] is None and h in provisional:
                p["real_C"], p["fuente_real"] = round(provisional[h], 2), "provisional"
                n_prov += 1

    registro.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas))
    print(f"verificado: {n_def} definitivos (ERA5), {n_prov} provisionales")

    juzgados = [p for f in filas for p in f["pasos"]
                if p["real_C"] is not None and p["oficial_C"] is not None]
    if juzgados:
        mae = lambda k: sum(abs(p[k] - p["real_C"]) for p in juzgados) / len(juzgados)
        print(f"marcador con {len(juzgados)} horas -> "
              f"IA local {mae('ia_local_C'):.2f} °C · oficial {mae('oficial_C'):.2f} °C")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="orden", required=True)

    r = sub.add_parser("registrar", help="guarda predicción local + oficial")
    r.add_argument("--pronostico", required=True, help="json que produce pronostico.py")
    r.add_argument("--registro", default="clima.jsonl")
    r.add_argument("--lugar", default="")
    r.add_argument("--modelo-oficial", default="ecmwf_ifs025",
                   help="modelo de Open-Meteo a usar como oficial")
    r.set_defaults(func=cmd_registrar)

    v = sub.add_parser("verificar", help="rellena lo que ya ocurrió")
    v.add_argument("--registro", default="clima.jsonl")
    v.add_argument("--lat", type=float, required=True)
    v.add_argument("--lon", type=float, required=True)
    v.set_defaults(func=cmd_verificar)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
