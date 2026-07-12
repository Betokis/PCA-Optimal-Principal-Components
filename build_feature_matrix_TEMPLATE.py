#!/usr/bin/env python3
"""
build_feature_matrix.py
=======================================================================
PLANTILLA DE TRABAJO -- IB Math AI SL

Objetivo del script (y SOLO este objetivo):

    Transformar 250 imágenes de dígitos (32x32, escala de grises) en
    una única matriz numérica

        X ∈ R^(250 x 1024)

    lista para ser usada como entrada del PCA en el siguiente script.

Este archivo NO calcula PCA, autovalores, autovectores ni nada de eso.
Solo construye X.

CÓMO USAR ESTA PLANTILLA
-------------------------------------------------------------------
Cada función tiene:
  1. Una docstring que explica QUÉ debe hacer (el contrato).
  2. Pistas en comentarios sobre qué funciones de Python/NumPy/Pillow
     puedes investigar (no se te da la línea de código ya escrita).
  3. Un `# TODO` marcando exactamente dónde falta tu código.

Ve completando las fases EN ORDEN (1 -> 9). Cada fase depende de que
la anterior funcione. Cuando termines una función, puedes probarla
sola antes de seguir (eso se llama "probar de forma incremental" y es
buena práctica profesional).
=======================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# =======================================================================
# FASE 1 — CONFIGURACIÓN DEL ENTORNO
# =======================================================================
# Concepto: separar "lo que puede cambiar" (rutas, tamaños) del resto
# del código, para que si mañana usas imágenes de 64x64 en vez de
# 32x32, solo cambies un número aquí arriba y no busques por todo el
# script.

# TODO 1.1: ruta a la carpeta donde están tus 250 PNG ya procesados
#           (la que generó el script anterior).
INPUT_DIR: Path = Path("???")  # ejemplo de forma: Path("Dataset_Procesado")

# TODO 1.2: ruta a la carpeta donde este script guardará X.npy y X.csv
OUTPUT_DIR: Path = Path("???")

# TODO 1.3: parámetros del dataset. Piensa: ¿cuánto mide cada imagen?
#           ¿cuántas imágenes esperas en total?
IMG_SIZE: int = 0          # <-- reemplaza 0 por el valor correcto
N_IMAGES_EXPECTED: int = 0  # <-- reemplaza 0 por el valor correcto

# Esta línea ya está resuelta para ti (es la relación matemática
# entre el tamaño de la imagen y la longitud del vector):
N_FEATURES: int = IMG_SIZE * IMG_SIZE  # 32*32 = 1024, si IMG_SIZE está bien


# =======================================================================
# FASE 2 — LECTURA Y VERIFICACIÓN DE UNA IMAGEN
# =======================================================================

def load_and_validate_image(path: Path, expected_size: int) -> np.ndarray:
    """Abre UNA imagen y verifica que sea válida para el análisis.

    Debe realizar, en este orden:
      1. Verificar que `path` existe. Si no, lanzar FileNotFoundError.
      2. Intentar abrir la imagen. Si falla (archivo corrupto), debe
         lanzar una excepción clara (piensa en try/except).
      3. Verificar que sus dimensiones sean (expected_size, expected_size).
         Si no lo son, lanzar ValueError con un mensaje útil.
      4. Verificar que esté en escala de grises. Si no lo está,
         convertirla (pista: Pillow representa el modo escala de
         grises con la letra 'L').

    Parameters
    ----------
    path : Path
        Ruta al archivo PNG individual (ej. 000.png).
    expected_size : int
        Ancho/alto esperado en píxeles (deben ser cuadradas).

    Returns
    -------
    np.ndarray
        Array 2D de forma (expected_size, expected_size), dtype uint8,
        con valores de intensidad entre 0 y 255.

    Pistas de herramientas a investigar:
      - `Path.exists()`
      - `PIL.Image.open(...)`
      - atributo `.size` de una imagen de Pillow (¡cuidado! Pillow
        devuelve (ancho, alto), no (alto, ancho))
      - atributo `.mode` de una imagen de Pillow ('L' = escala de
        grises de 8 bits)
      - método `.convert('L')`
      - `np.array(imagen_de_pillow)` para pasar de objeto Pillow a
        array de NumPy
    """
    # TODO 2.1: verificar existencia del archivo

    # TODO 2.2: abrir la imagen (maneja el caso de archivo corrupto)

    # TODO 2.3: verificar tamaño == (expected_size, expected_size)

    # TODO 2.4: verificar/forzar escala de grises

    # TODO 2.5: convertir a np.ndarray y hacer return
    raise NotImplementedError("Completa load_and_validate_image")


# =======================================================================
# FASE 3 — NORMALIZACIÓN
# =======================================================================

def normalize_image(img_array: np.ndarray) -> np.ndarray:
    """Reescala los valores de intensidad del rango [0, 255] al rango
    [0, 1].

    Pregunta para ti antes de programar: si divides un array de dtype
    `uint8` entre 255 usando división entera, ¿qué obtienes? ¿Qué
    necesitas hacer con el `dtype` del array ANTES de dividir para
    obtener decimales?

    Parameters
    ----------
    img_array : np.ndarray
        Array 2D, dtype uint8, valores en [0, 255].

    Returns
    -------
    np.ndarray
        Array 2D, dtype float, valores en [0.0, 1.0].

    Pista: `.astype(np.float32)` o `.astype(np.float64)`.
    """
    # TODO 3.1: convertir el dtype y dividir entre el valor máximo posible
    raise NotImplementedError("Completa normalize_image")


# =======================================================================
# FASE 4 — VECTORIZACIÓN
# =======================================================================

def vectorize_image(img_array: np.ndarray) -> np.ndarray:
    """Convierte una imagen 2D (IMG_SIZE x IMG_SIZE) en un vector 1D
    de longitud N_FEATURES, SIN perder ni reordenar información: solo
    cambia la forma en que está almacenada.

    Parameters
    ----------
    img_array : np.ndarray
        Array 2D normalizado, forma (IMG_SIZE, IMG_SIZE).

    Returns
    -------
    np.ndarray
        Array 1D, forma (N_FEATURES,).

    Pista: busca en la documentación de NumPy los métodos `.flatten()`
    y `.reshape(-1)`. Investiga la diferencia entre ellos (una de las
    dos formas crea una COPIA y la otra a veces no; para este script
    no importa cuál elijas, pero es bueno que sepas la diferencia).
    """
    # TODO 4.1: aplanar la matriz a un vector 1D
    raise NotImplementedError("Completa vectorize_image")


# =======================================================================
# FASE 5 — CONSTRUCCIÓN DE LA MATRIZ X
# =======================================================================

def build_matrix(vectors: List[np.ndarray]) -> np.ndarray:
    """Apila una lista de vectores (cada uno de longitud N_FEATURES)
    como filas de una única matriz 2D.

    Parameters
    ----------
    vectors : List[np.ndarray]
        Lista de longitud N_IMAGES_EXPECTED, cada elemento un array 1D
        de forma (N_FEATURES,).

    Returns
    -------
    np.ndarray
        Matriz X de forma (N_IMAGES_EXPECTED, N_FEATURES).

    Pista: investiga `np.stack` o `np.array(lista_de_vectores)`.
    Piensa qué eje (axis) corresponde a "una imagen por fila".
    """
    # TODO 5.1: construir y devolver la matriz X
    raise NotImplementedError("Completa build_matrix")


# =======================================================================
# FASE 6 — VALIDACIÓN MATEMÁTICA DE X
# =======================================================================

def validate_matrix(X: np.ndarray, expected_shape: Tuple[int, int]) -> None:
    """Comprueba que la matriz construida sea matemáticamente
    consistente ANTES de guardarla o de pasarla al PCA.

    Debe verificar (y avisar con un mensaje claro si algo falla):
      1. X.shape == expected_shape
      2. Todos los valores de X están en el intervalo [0, 1]
      3. X no contiene NaN ni infinitos (pista: `np.isnan`, `np.isinf`)

    No hace falta que lances excepciones para todo: puedes usar
    `assert condicion, "mensaje"` como forma rápida de validar
    supuestos durante el desarrollo.

    Parameters
    ----------
    X : np.ndarray
        La matriz a validar.
    expected_shape : Tuple[int, int]
        La forma que X DEBERÍA tener, ej. (250, 1024).
    """
    # TODO 6.1: verificar la forma (shape)
    # TODO 6.2: verificar el rango de valores
    # TODO 6.3: verificar ausencia de NaN/inf
    raise NotImplementedError("Completa validate_matrix")


# =======================================================================
# FASE 7 — ESTADÍSTICAS DESCRIPTIVAS
# =======================================================================

def compute_statistics(X: np.ndarray) -> Dict[str, float]:
    """Calcula estadísticas descriptivas globales sobre TODOS los
    valores de X (no por fila ni por columna, sino sobre la matriz
    completa).

    Returns
    -------
    Dict[str, float]
        Diccionario con, como mínimo, las claves:
        "mean", "std", "min", "max".

    Pista: los arrays de NumPy tienen métodos `.mean()`, `.std()`,
    `.min()`, `.max()` que operan sobre todos los elementos si no les
    pasas un argumento `axis`.
    """
    # TODO 7.1: calcular y devolver el diccionario de estadísticas
    raise NotImplementedError("Completa compute_statistics")


# =======================================================================
# FASE 8 — GUARDADO
# =======================================================================

def save_matrix(X: np.ndarray, output_dir: Path) -> None:
    """Guarda la matriz X en dos formatos:
      - `X.npy`  (formato binario nativo de NumPy: rápido, exacto,
                  ideal para que el script de PCA lo vuelva a cargar)
      - `X.csv`  (formato de texto plano: para que TÚ puedas abrirlo
                  en Excel/Sheets y revisarlo a simple vista)

    Parameters
    ----------
    X : np.ndarray
        La matriz ya validada.
    output_dir : Path
        Carpeta donde guardar ambos archivos (créala si no existe).

    Pistas: `output_dir.mkdir(parents=True, exist_ok=True)`,
    `np.save(...)`, `np.savetxt(..., delimiter=",")`.
    """
    # TODO 8.1: crear la carpeta de salida si no existe
    # TODO 8.2: guardar X.npy
    # TODO 8.3: guardar X.csv
    raise NotImplementedError("Completa save_matrix")


# =======================================================================
# FASE 9 — VISUALIZACIÓN
# =======================================================================

def plot_vectorization_example(
    img_array: np.ndarray, vector: np.ndarray, save_path: Path
) -> None:
    """Figura 1: muestra, lado a lado, (a) la imagen original como
    imagen, (b) esa misma imagen como matriz de números (puedes
    mostrar la matriz con una tabla o con `imshow` + anotaciones), y
    (c) el vector resultante tras aplanarla.

    El objetivo pedagógico de esta figura es que cualquiera que la
    vea entienda visualmente en qué consiste "vectorizar".

    Pistas: `plt.subplots(1, 3, figsize=(...))`, `ax.imshow(...)`,
    `ax.plot(vector)` o representar el vector como una franja de 1x1024
    con `imshow` también.
    """
    # TODO 9.1: implementar la figura de ejemplo de vectorización
    raise NotImplementedError("Completa plot_vectorization_example")


def plot_matrix_heatmap(X: np.ndarray, save_path: Path) -> None:
    """Figura 2: mapa de calor (heatmap) de la matriz X completa.

    Piensa qué debe ir en cada eje: ¿qué representan las filas de X?
    ¿Qué representan las columnas? Tu heatmap debería dejar eso claro
    con etiquetas de eje (`ax.set_xlabel`, `ax.set_ylabel`).

    Pista: `plt.imshow(X, aspect='auto', cmap=...)`, `plt.colorbar()`.
    """
    # TODO 9.2: implementar el heatmap de X
    raise NotImplementedError("Completa plot_matrix_heatmap")


def plot_intensity_histogram(X: np.ndarray, save_path: Path) -> None:
    """Figura 3: histograma de TODOS los valores de intensidad
    contenidos en X (250 x 1024 = 256000 valores), para ver cómo se
    distribuyen entre 0 y 1.

    Pista: necesitas aplanar X a un vector 1D antes de pasarlo a
    `plt.hist(...)` (reutiliza lo que aprendiste en la Fase 4).
    """
    # TODO 9.3: implementar el histograma de intensidades
    raise NotImplementedError("Completa plot_intensity_histogram")


# =======================================================================
# ORQUESTACIÓN PRINCIPAL
# =======================================================================

def main() -> None:
    """Conecta todas las fases en orden. Esta función ya está casi
    completa como guía de la SECUENCIA correcta -- tu trabajo es
    llenar los huecos marcados, usando las funciones que programaste
    arriba."""

    print("FASE 1: configuración cargada.")
    print(f"  INPUT_DIR  = {INPUT_DIR}")
    print(f"  OUTPUT_DIR = {OUTPUT_DIR}")
    print(f"  IMG_SIZE   = {IMG_SIZE}  ->  N_FEATURES = {N_FEATURES}")

    # TODO M.1 (Fase 2): construye la lista ordenada de rutas a las
    # 250 imágenes (000.png ... 249.png) dentro de INPUT_DIR.
    # Pista: `sorted(INPUT_DIR.glob("*.png"))` -- ¿por qué hace falta
    # `sorted()` aquí? ¿Qué pasaría si no lo usaras?
    image_paths: List[Path] = []  # TODO: reemplaza esta línea

    # TODO M.2 (Fases 2-4): para cada ruta en image_paths...
    #   a) cárgala y valídala con load_and_validate_image
    #   b) normalízala con normalize_image
    #   c) vectorízala con vectorize_image
    #   d) guarda el resultado en una lista `vectors`
    vectors: List[np.ndarray] = []  # TODO: llena esta lista con un bucle for

    # TODO M.3 (Fase 5): construye X a partir de `vectors`
    X: np.ndarray = None  # TODO: reemplaza con build_matrix(vectors)

    # TODO M.4 (Fase 6): valida X contra la forma esperada
    # (N_IMAGES_EXPECTED, N_FEATURES)

    # TODO M.5 (Fase 7): calcula estadísticas descriptivas e imprímelas
    # por pantalla de forma clara (usa print con f-strings)

    # TODO M.6 (Fase 8): guarda X en OUTPUT_DIR

    # TODO M.7 (Fase 9): genera las 3 figuras
    #   - para la Figura 1, elige UNA imagen de ejemplo (ej. la primera)
    #     y su vector correspondiente

    print("\nListo. Revisa OUTPUT_DIR para ver X.npy, X.csv y las figuras.")


if __name__ == "__main__":
    main()
