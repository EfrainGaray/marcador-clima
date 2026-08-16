# Marcador del clima

Corre el modelo de pronóstico de ECMWF en tu propia GPU, guarda lo que predice
junto a lo que predice el pronóstico oficial, y después anota lo que de verdad
ocurrió.

La idea es simple: las predicciones se escriben **antes** del hecho y no se
tocan nunca más. Solo se añade la columna de la realidad. Con los meses, el
registro dice quién acierta más sin que haya que creerle a nadie.

Marcador en vivo: **[efraingaray.com/clima](https://efraingaray.com/clima/)** ·
Cómo se hizo: **[el artículo](https://efraingaray.com/blog/aifs-pronostico-casa/)**

## Qué mide

| | |
|---|---|
| **IA local** | AIFS Single 2.0 de ECMWF, pesos abiertos, corriendo en tu máquina |
| **Oficial** | IFS operativo de ECMWF, el modelo físico que corre en superordenador |
| **Real** | Reanálisis ERA5; mientras llega, el análisis reciente marcado como provisional |

## Lo que cuesta correrlo

Medido en una RTX 4070 Ti SUPER de 16 GB, pronóstico a 48 horas:

| | GPU | CPU (16 hilos) |
|---|---|---|
| Tiempo | 108.5 s | 1984.3 s |
| Memoria de video | 5.77 GB | — |
| Punto de control | 0.93 GB | 0.93 GB |

Las dos rutas dan el mismo resultado: la diferencia media entre GPU y CPU fue
de 0.0075 °C, y aparece recién en el sexto paso, porque cada paso toma como
entrada la salida del anterior y el error numérico se acumula.

Cabe de sobra en una tarjeta de 8 GB. Sin tarjeta también funciona, con
paciencia.

## Instalación

Necesitas Python 3.11 o 3.12, y el envoltorio de Hugging Face que sustituye
`flash-attn` por la atención propia de PyTorch (`flash-attn` solo compila en
tarjetas Ampere o más nuevas):

```bash
mkdir -p ~/aifs && cd ~/aifs
python3.12 -m venv venv
git clone https://github.com/huggingface/AIFS-single-2.0-on-all-GPUs
cd AIFS-single-2.0-on-all-GPUs
~/aifs/venv/bin/pip install -r requirements.txt
git clone https://github.com/EfrainGaray/marcador-clima
cp marcador-clima/*.py .
```

**No instales `flash-attn`.** Ese es todo el truco del envoltorio.

### El tropiezo que cuesta una tarde

El `requirements.txt` pide `torch>=2.1` y pip resuelve eso a la última versión,
que puede venir compilada contra una CUDA más nueva que tu driver. PyTorch no
falla: **desactiva la GPU y sigue en CPU sin decir nada.** Comprueba siempre:

```bash
~/aifs/venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Si sale `False` y tienes tarjeta, instala la build que corresponde a tu driver
(por ejemplo, para CUDA 12.8):

```bash
~/aifs/venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.7.*"
```

## Uso

**1. Pronosticar.** Las condiciones iniciales salen del portal de datos
abiertos de ECMWF, gratis y sin cuenta. La primera descarga tarda unos minutos
y queda en caché.

```bash
python pronostico.py --lat -33.4489 --lon -70.6693 --horas 48 --salida hoy.json
```

Con `--cpu` fuerza procesador. Con `--trozos` (por defecto 16) subes el número
si te quedas sin memoria de video.

**2. Registrar** la predicción propia junto a la oficial:

```bash
python marcador.py registrar --pronostico hoy.json --registro clima.jsonl \
                             --lugar "Santiago de Chile"
```

**3. Verificar**, cuando el tiempo ya pasó:

```bash
python marcador.py verificar --registro clima.jsonl --lat -33.4489 --lon -70.6693
```

Imprime el marcador acumulado:

```
verificado: 3 definitivos (ERA5), 0 provisionales
marcador con 3 horas
  IA local : MAE 0.44 °C · RMSE 0.52 °C
  oficial  : MAE 0.33 °C · RMSE 0.41 °C
```

El **MAE** es el error medio absoluto. El **RMSE** eleva al cuadrado antes de
promediar, así que castiga más los errores grandes: si queda bastante por encima
del MAE, el modelo falla poco pero falla feo. En clima eso importa, porque lo que
duele son los extremos.

**4. Exportar** el registro a CSV, con los errores ya calculados:

```bash
python marcador.py exportar --registro clima.jsonl --salida clima.csv
```

Una fila por paso de pronóstico, con el horizonte en horas y ambos errores. Las
horas que todavía no ocurren van con la columna real vacía: nunca se rellenan con
una estimación, porque entonces el archivo dejaría de servir para juzgar.

Los pasos 1 a 3 encadenados, una vez al día, es todo lo que hace falta.

## El formato del registro

Un JSONL, una línea por corrida:

```json
{
  "analisis_utc": "2026-08-16 00:00:00",
  "lugar": "Santiago de Chile",
  "hardware": "NVIDIA GeForce RTX 4070 Ti SUPER",
  "segundos_pronostico": 108.5,
  "pasos": [
    {"hora_utc": "2026-08-16T06:00", "ia_local_C": 5.77,
     "oficial_C": 6.3, "real_C": 6.6, "fuente_real": "era5"}
  ]
}
```

`verificar` solo escribe `real_C` y `fuente_real`. Un valor provisional se
reemplaza por el definitivo de ERA5; uno definitivo no se vuelve a tocar. Las
dos columnas de predicción no se editan jamás: si se editaran, el registro no
valdría nada.

## Detalles que importan

**La grilla no es rectangular.** AIFS entrega una grilla gaussiana reducida, un
vector de puntos con sus coordenadas. Para una ciudad se toma el punto más
cercano, que puede caer a diez o quince kilómetros. No hay interpolación.

**`msl` no es presión de superficie.** El modelo entrega presión reducida a
nivel del mar. Compararla contra la presión de superficie de una API da
diferencias enormes que parecen un error del modelo y no lo son: en una ciudad
a 570 metros, la diferencia esperada es de unos 65 hPa.

**La atención no es la original.** El envoltorio sustituye `flash-attn` por
SDPA, y avisa de que los resultados no son bit a bit idénticos a los del
entrenamiento. Parte del error medido es de esa sustitución, no del modelo.

**El IFS se mueve.** Se actualiza cuatro veces al día, y si consultas el
pronóstico oficial vigente puede venir de un análisis más reciente que el que usó
tu corrida local, con horas extra de información sobre la atmósfera. Eso favorece
al oficial. Para una comparación estricta hay que fijar ambas al mismo análisis
inicial; este código todavía no lo hace, y conviene saberlo al leer los números.

**No es para uso operativo.** El propio envoltorio lo dice. Es un modelo
determinista, así que da un único futuro y no una distribución; para eso está
`aifs-ens-2.0`. Y tiende a suavizar los extremos, que es justo lo que más
importa cuando el clima importa de verdad.

## Licencia

MIT para este código. Los pesos de AIFS son de ECMWF, con licencia CC-BY-4.0.
Este repositorio no está afiliado a ECMWF.
