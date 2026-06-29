# UNIR-TFM

Código asociado al TFM **“Aprendizaje automático aplicado a la detección temprana de flujos migratorios”**.

El proyecto combina dos partes:

1. **Extracción de señales desde Google Trends** con una lista de palabras clave relacionada con la temática del trabajo.
2. **Clasificación y comparación de modelos** sobre un dataset tabular, usando **CART** y una red neuronal tipo **MLP**.

## Qué hay en el repo

| Archivo | Qué hace |
|---|---|
| `trends.py` | Consulta Google Trends con `pytrends` y genera `search_trends_SN.csv`. |
| `cart_complex_2506.py` | Carga `datos_complex.csv`, codifica variables categóricas, entrena un árbol CART y guarda `CARTdecisiontree.png`. |
| `optimizing.py` | Prueba distintas configuraciones de una red neuronal con Keras/TensorFlow. |
| `datos_complex.csv` | Dataset principal usado por los scripts de ML. |
| `keyword_list.csv` | Lista de palabras clave para las consultas de Trends. |
| `optimizacion.jpg`, `optimizacion2.jpg`, `CARTdecisiontree.png` | Salidas/gráficos generados por los scripts. |

## Contexto del TFM

El trabajo busca predecir posibles eventos migratorios futuros en la frontera sur española combinando:

- datos de cruces de frontera,
- datos de búsquedas en internet en los países de origen,
- y modelos de aprendizaje automático.

Según la memoria, el modelo **CART** ofrece mejores resultados e interpretación que la red neuronal usada en la comparación.

## Requisitos

El proyecto está pensado para **Python 3** y usa, como mínimo:

- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `pydotplus`
- `pytrends`

> El repo ya incluye un `requirements.txt` mínimo. Conviene usar un entorno virtual.

## Ejecución

### Google Trends

```bash
python trends.py
```

Salida esperada:

- `search_trends_SN.csv`

### Árbol CART

```bash
python cart_complex_2506.py
```

Salida esperada:

- métricas por consola
- `CARTdecisiontree.png`

### Optimización de la red neuronal

```bash
python optimizing.py
```

Puede tardar bastante porque prueba varias combinaciones de capas y parámetros.

## Datos

- `datos_complex.csv` se lee con separador `;`.
- Las variables categóricas (`NACIONALIDAD`, `RUTA`, `TIPO`, `FMM`) se convierten a números con `LabelEncoder`.
- `trends.py` consulta el geo `SN`.
- La memoria del TFM indica que el dataset final tiene **4.416 registros**.

## Estructura rápida

```text
.
├── cart_complex_2506.py
├── optimizing.py
├── trends.py
├── datos_complex.csv
├── keyword_list.csv
├── CARTdecisiontree.png
└── LICENSE
```

## Licencia

Este proyecto está bajo **GPL-2.0**. Consulta `LICENSE` para los términos completos.
