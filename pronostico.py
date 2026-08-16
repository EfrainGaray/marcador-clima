#!/usr/bin/env python3
"""Corre AIFS Single 2.0 y extrae el pronóstico para un punto de la Tierra.

AIFS es el modelo de pronóstico basado en datos de ECMWF. Sus pesos son
públicos (CC-BY-4.0) y el punto de control pesa menos de un giga, así que corre
en una GPU de escritorio: 48 horas de pronóstico en unos 108 segundos y 5.77 GB
de memoria en una RTX 4070 Ti SUPER. También corre en CPU, unas 18 veces más
lento, con resultados que coinciden hasta la centésima de grado.

Necesita el envoltorio de Hugging Face que sustituye flash-attn por la atención
propia de PyTorch, porque flash-attn solo compila en tarjetas Ampere o más
nuevas:

    git clone https://github.com/huggingface/AIFS-single-2.0-on-all-GPUs

Este script se ejecuta desde dentro de ese repositorio.

Uso:
    python pronostico.py --lat -33.4489 --lon -70.6693 --horas 48 \
                         --salida pronostico.json
"""

import argparse
import json
import time

import numpy as np
import torch

from aifs import load_ics, run_forecast


def punto_mas_cercano(lats, lons, lat, lon):
    """Índice del punto de grilla más cercano.

    AIFS entrega una grilla gaussiana reducida, es decir un vector de puntos con
    sus coordenadas, no una malla rectangular. Para una ciudad se toma el punto
    más cercano: no hay interpolación.
    """
    lon = lon % 360
    glon = np.asarray(lons) % 360
    glat = np.asarray(lats)
    dlon = np.minimum(np.abs(glon - lon), 360 - np.abs(glon - lon))
    return int(np.argmin(np.hypot(glat - lat, dlon * np.cos(np.radians(lat)))))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--horas", type=int, default=48, help="horizonte, múltiplo de 6")
    p.add_argument("--trozos", type=int, default=16,
                   help="parte el grafo para bajar la memoria; súbelo si te quedas sin VRAM")
    p.add_argument("--cache", default="ic_cache", help="dónde guardar las condiciones iniciales")
    p.add_argument("--salida", default="pronostico.json")
    p.add_argument("--cpu", action="store_true", help="fuerza CPU aunque haya GPU")
    args = p.parse_args()

    usa_gpu = torch.cuda.is_available() and not args.cpu
    equipo = torch.cuda.get_device_name(0) if usa_gpu else "CPU"
    print(f"dispositivo: {equipo}")
    if not usa_gpu and torch.cuda.is_available():
        print("aviso: hay GPU disponible pero se pidió CPU")
    elif not torch.cuda.is_available():
        # El fallo silencioso más común: torch instalado para una versión de CUDA
        # más nueva que el driver. PyTorch no falla, solo desactiva la GPU.
        print("aviso: CUDA no disponible. Si tienes tarjeta, comprueba que la build "
              "de torch coincida con tu driver (por ejemplo cu128 para CUDA 12.8).")
    if usa_gpu:
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    campos, fecha = load_ics(cache_dir=args.cache)
    t_ics = time.perf_counter() - t0
    print(f"condiciones iniciales: {fecha} ({t_ics:.1f}s)")

    t1 = time.perf_counter()
    estados = run_forecast(campos, fecha, lead_time=args.horas, num_chunks=args.trozos)
    t_fc = time.perf_counter() - t1
    print(f"pronóstico de {args.horas} h en {t_fc:.1f}s "
          f"({len(estados)} pasos, {t_fc/max(1,len(estados)):.1f}s por paso)")

    vram = torch.cuda.max_memory_allocated() / 2**30 if usa_gpu else 0.0
    if vram:
        print(f"memoria de video, pico: {vram:.2f} GB")

    e0 = estados[0]
    if e0.get("latitudes") is None:
        raise SystemExit(f"el estado no trae coordenadas; claves: {list(e0.keys())}")
    idx = punto_mas_cercano(e0["latitudes"], e0["longitudes"], args.lat, args.lon)
    glat = float(np.asarray(e0["latitudes"])[idx])
    glon = float(np.asarray(e0["longitudes"])[idx])
    glon = glon - 360 if glon > 180 else glon
    print(f"punto de grilla más cercano: {glat:.3f}, {glon:.3f}")

    pasos = []
    for st in estados:
        f = st["fields"]
        fila = {"fecha_utc": str(st["date"])}
        if "2t" in f:                       # kelvin -> celsius
            fila["t2m_C"] = round(float(np.asarray(f["2t"])[idx]) - 273.15, 2)
        if "msl" in f:                      # ojo: presión a NIVEL DEL MAR, no de superficie
            fila["presion_msl_hPa"] = round(float(np.asarray(f["msl"])[idx]) / 100, 1)
        if "10u" in f and "10v" in f:
            u = float(np.asarray(f["10u"])[idx])
            v = float(np.asarray(f["10v"])[idx])
            fila["viento_ms"] = round((u**2 + v**2) ** 0.5, 2)
        pasos.append(fila)
        print(f"  {fila['fecha_utc']}  {fila.get('t2m_C','?'):>6} °C")

    salida = {
        "modelo": "ECMWF AIFS Single 2.0",
        "advertencia": "flash-attn sustituido por SDPA; no apto para uso operativo",
        "hardware": equipo,
        "coords_pedidas": [args.lat, args.lon],
        "coords_grilla": [glat, glon],
        "analisis_inicial_utc": str(fecha),
        "segundos_carga_ics": round(t_ics, 1),
        "segundos_pronostico": round(t_fc, 1),
        "vram_pico_GB": round(vram, 2),
        "pasos": pasos,
    }
    with open(args.salida, "w") as fh:
        json.dump(salida, fh, indent=2, ensure_ascii=False)
    print(f"-> {args.salida}")


if __name__ == "__main__":
    main()
