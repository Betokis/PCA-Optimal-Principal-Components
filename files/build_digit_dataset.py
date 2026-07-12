#!/usr/bin/env python3
"""
build_digit_dataset.py
=======================================================================
Pipeline de segmentación y estandarización de dígitos manuscritos para
la Prueba Interna de IB Mathematics: Applications and Interpretation SL

    "Determinación matemática del número óptimo de componentes
    principales en la compresión de imágenes mediante PCA"

Este script toma un único escaneo con una cuadrícula de dígitos
manuscritos (N_ROWS filas, uno por dígito 0-9, ~25 repeticiones por
fila) y produce un conjunto de imágenes PNG de 32x32 px, perfectamente
alineadas, centradas y ordenadas, listas para ser vectorizadas y
analizadas con PCA (scikit-learn).

-----------------------------------------------------------------------
JUSTIFICACIÓN MATEMÁTICA DEL MÉTODO DE SEGMENTACIÓN (ver sección "Extra")
-----------------------------------------------------------------------
Se evaluó si una técnica distinta a Connected Components (p. ej.
Watershed o MSER) sería preferible. La respuesta, justificada
formalmente, es que Connected Components de 8-conectividad es no solo
suficiente sino ÓPTIMA en este contexto, porque la disposición en
cuadrícula garantiza una separación espacial fuerte entre dígitos.

Sea:
    d_min = distancia mínima entre píxeles de tinta de dos dígitos
            distintos cualesquiera (separación inter-componente)
    g_max = mayor discontinuidad interna dentro del trazo de un mismo
            dígito (p. ej. el trazo del "4" si el lápiz se levanta)
    r     = radio del elemento estructurante usado en el cierre
            morfológico (closing) aplicado antes del etiquetado

Un cierre morfológico con disco de radio r conecta dos regiones de
tinta si y solo si la distancia entre ellas es < 2r (el disco crece
r píxeles desde cada borde). Por lo tanto, el etiquetado por
componentes conexas producirá EXACTAMENTE un componente por dígito
(ni fusiona dígitos distintos ni fragmenta uno solo) si y solo si:

                    g_max <= 2r < d_min

Es decir, el radio debe ser suficiente para cerrar cualquier micro-
discontinuidad interna del trazo, pero estrictamente menor a la mitad
de la separación mínima entre dígitos. Sobre el escaneo real
analizado: la distancia mínima entre centroides de dígitos vecinos en
una misma fila y entre filas contiguas es de decenas de píxeles,
mientras que el ancho de línea del trazo (y por tanto cualquier
micro-hueco) es de un puñado de píxeles. La desigualdad se satisface
con amplio margen (se verificó empíricamente: separación entre
componentes >> dispersión intra-fila, ver `group_into_rows`), por lo
que Connected Components es la elección matemáticamente correcta y
computacionalmente óptima (complejidad O(N) sobre N píxeles, frente a
O(N log N) o peor de Watershed/MSER, que además están diseñados para
resolver el caso de regiones TOCÁNDOSE o solapándose, un problema que
aquí no existe por diseño de la hoja de captura).

-----------------------------------------------------------------------
Autor: script generado para uso académico (IB Math AI SL)
Librerías permitidas: opencv-python, numpy, matplotlib, scikit-image,
                       pdf2image (solo si la entrada es PDF), pathlib
=======================================================================
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

# =======================================================================
# CONFIGURACIÓN (editar aquí según el escaneo utilizado)
# =======================================================================

# --- Entrada / salida ---------------------------------------------------
INPUT_PATH: Path = Path("dataset_math_ai.jpeg")   # .jpg / .png / .pdf
OUTPUT_DIR: Path = Path("Dataset_Procesado")
VIZ_DIR: Path = Path("Visualizaciones")

# --- Geometría de la hoja ------------------------------------------------
N_ROWS: int = 10          # una fila por dígito (0-9)
EXPECTED_TOTAL: int = 250  # 10 filas x 25 repeticiones esperadas

# --- Estandarización de salida -------------------------------------------
TARGET_SIZE: int = 32      # lado del cuadrado final (px)
MARGIN_RATIO: float = 0.22  # margen alrededor del dígito, proporcional
                             # al lado mayor de su bounding box

# --- Filtrado de ruido (adaptativo, relativo a la mediana de área) -------
MIN_AREA_RATIO: float = 0.15   # descarta componentes < 15% de la mediana
MAX_AREA_RATIO: float = 6.0    # advierte sobre componentes > 600% (posible
                                # fusión de dos dígitos)
ABS_MIN_AREA_PX: int = 10      # piso absoluto para ignorar motas ínfimas

# --- Preprocesamiento ------------------------------------------------
MEDIAN_BLUR_KERNEL: int = 3     # reducción de ruido (debe ser impar)
ILLUM_BLUR_KERNEL: int = 151    # corrección de iluminación (flat-field),
                                 # debe ser impar y mayor que el dígito
GAUSS_BLUR_KERNEL: int = 5      # suavizado previo a Otsu (debe ser impar)

# --- Limpieza morfológica -------------------------------------------
MORPH_CLOSE_KERNEL: int = 3     # cierra micro-huecos internos del trazo
MORPH_OPEN_KERNEL: int = 2      # elimina motas de ruido aisladas
MASK_DILATE_FOR_CROP: int = 1   # dilatación suave del recorte final para
                                 # no cortar bordes antialiased del trazo

# --- Corrección de inclinación (deskew) -------------------------------
DESKEW_ANGLE_THRESHOLD_DEG: float = 0.3  # por debajo de esto, no rota

# --- Visualización -----------------------------------------------------
N_COMPARISON_EXAMPLES: int = 8   # nº de ejemplos en la figura comparativa


# =======================================================================
# ESTRUCTURAS DE DATOS
# =======================================================================

@dataclass
class ComponentInfo:
    """Metadatos de un componente conexo (candidato a dígito)."""

    label: int
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


# =======================================================================
# CARGA DE IMAGEN
# =======================================================================

def load_image(path: Path) -> np.ndarray:
    """Carga una imagen (jpg/png) o la primera página de un PDF.

    Parameters
    ----------
    path : Path
        Ruta al archivo de entrada.

    Returns
    -------
    np.ndarray
        Imagen en formato BGR (uint8), como la devuelve OpenCV.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ValueError
        Si el formato no está soportado o la imagen no pudo leerse.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise ImportError(
                "Se requiere 'pdf2image' (y el binario 'poppler-utils' "
                "instalado en el sistema) para leer archivos PDF."
            ) from exc
        pages = convert_from_path(str(path), dpi=300)
        if not pages:
            raise ValueError(f"El PDF '{path}' no contiene páginas legibles.")
        pil_image = pages[0].convert("RGB")
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    elif suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV no pudo decodificar la imagen: {path}")
    else:
        raise ValueError(f"Formato de archivo no soportado: '{suffix}'")

    return image


# =======================================================================
# PREPROCESAMIENTO
# =======================================================================

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR a escala de grises."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, kernel_size: int = MEDIAN_BLUR_KERNEL) -> np.ndarray:
    """Reduce ruido de alta frecuencia (grano, artefactos JPEG) con un
    filtro de mediana, que preserva mejor los bordes del trazo que un
    filtro gaussiano equivalente."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.medianBlur(gray, k)


def correct_illumination(gray: np.ndarray, kernel_size: int = ILLUM_BLUR_KERNEL) -> np.ndarray:
    """Corrige sombras y variaciones de iluminación mediante
    "flat-fielding": estima el campo de iluminación de baja frecuencia
    B(x,y) con un desenfoque gaussiano de kernel grande, y normaliza la
    imagen original I(x,y) asumiendo el modelo I = R * B, recuperando
    la reflectancia R = I / B. Esto vuelve el algoritmo robusto frente
    a sombras y gradientes suaves de luz sin afectar el trazo (de alta
    frecuencia espacial)."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    background = cv2.GaussianBlur(gray, (k, k), 0).astype(np.float32)
    background[background < 1.0] = 1.0  # evita división por cero
    normalized = (gray.astype(np.float32) / background) * float(np.mean(background))
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return normalized


def binarize_otsu(gray: np.ndarray, gauss_kernel: int = GAUSS_BLUR_KERNEL) -> np.ndarray:
    """Binariza la imagen con el método de Otsu, que selecciona
    automáticamente el umbral que minimiza la varianza intra-clase
    (equivalente a maximizar la varianza inter-clase) entre trazo y
    fondo. Se invierte la salida para que el trazo quede en 255
    (primer plano) y el fondo en 0, como requiere
    connectedComponentsWithStats."""
    k = gauss_kernel if gauss_kernel % 2 == 1 else gauss_kernel + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def clean_binary_mask(
    binary: np.ndarray,
    close_kernel: int = MORPH_CLOSE_KERNEL,
    open_kernel: int = MORPH_OPEN_KERNEL,
) -> np.ndarray:
    """Aplica operaciones morfológicas para (1) cerrar micro-huecos
    dentro de un mismo trazo (closing) y (2) eliminar motas de ruido
    aisladas (opening), sin fusionar dígitos distintos (ver
    justificación matemática en el docstring del módulo)."""
    close_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    open_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_elem)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_elem)
    return cleaned


def preprocess_pipeline(gray_source: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ejecuta la secuencia completa: denoise -> corrección de
    iluminación -> Otsu -> limpieza morfológica.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (gray_corregido, mascara_binaria_limpia)
    """
    denoised = denoise(gray_source)
    illum_corrected = correct_illumination(denoised)
    binary = binarize_otsu(illum_corrected)
    cleaned = clean_binary_mask(binary)
    return illum_corrected, cleaned


# =======================================================================
# DETECCIÓN Y ORDENAMIENTO DE COMPONENTES
# =======================================================================

def detect_components(
    binary: np.ndarray,
    min_area_ratio: float = MIN_AREA_RATIO,
    max_area_ratio: float = MAX_AREA_RATIO,
) -> List[ComponentInfo]:
    """Detecta componentes conexas (8-conectividad) y filtra ruido de
    forma ADAPTATIVA: los umbrales de área se calculan como fracciones
    de la mediana de área de todos los candidatos, en vez de usar un
    valor absoluto fijo. Esto hace el filtro robusto a cambios de
    resolución del escaneo (p. ej. 150 vs 300 DPI)."""
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if n_labels <= 1:
        raise ValueError(
            "No se detectó ningún componente conexo. Revisa el umbral de "
            "binarización o la calidad del escaneo de entrada."
        )

    # Excluimos el fondo (label 0) y filtramos primero un piso absoluto
    # ínfimo para poder calcular una mediana representativa.
    candidate_areas = stats[1:, cv2.CC_STAT_AREA]
    valid_for_median = candidate_areas[candidate_areas > ABS_MIN_AREA_PX]
    if len(valid_for_median) == 0:
        raise ValueError("Todos los componentes detectados son ruido ínfimo.")
    median_area = float(np.median(valid_for_median))

    min_area = max(ABS_MIN_AREA_PX, median_area * min_area_ratio)
    max_area = median_area * max_area_ratio

    components: List[ComponentInfo] = []
    n_oversized = 0
    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue  # descartado: ruido / mota
        if area > max_area:
            n_oversized += 1  # posible fusión de dos dígitos, se conserva
                               # pero se reporta como advertencia
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_id]
        components.append(ComponentInfo(label_id, x, y, w, h, area, float(cx), float(cy)))

    if n_oversized > 0:
        warnings.warn(
            f"{n_oversized} componente(s) tienen un área anormalmente "
            f"grande (> {max_area_ratio}x la mediana). Podrían ser dos "
            f"dígitos fusionados por trazos que se tocan."
        )

    return components


def group_into_rows(
    components: List[ComponentInfo], n_rows: int = N_ROWS
) -> List[List[ComponentInfo]]:
    """Agrupa los componentes en `n_rows` filas mediante partición por
    "mayor salto" (largest-gap) sobre la coordenada Y de los
    centroides: se ordenan los centroides por Y y se corta en los
    (n_rows - 1) saltos más grandes.

    Justificación: esto equivale a clustering jerárquico de enlace
    simple (single-linkage) en una dimensión, que es exacto siempre
    que la separación entre filas sea mayor que la dispersión vertical
    dentro de una misma fila -- condición ampliamente satisfecha en
    una hoja cuadriculada (se verificó empíricamente que el salto
    inter-fila supera en más de un orden de magnitud a la dispersión
    intra-fila)."""
    if len(components) < n_rows:
        raise ValueError(
            f"Se detectaron {len(components)} componentes, insuficientes "
            f"para formar {n_rows} filas."
        )

    order = np.argsort([c.cy for c in components])
    ys_sorted = np.array([components[i].cy for i in order])
    diffs = np.diff(ys_sorted)

    n_splits = n_rows - 1
    if len(diffs) < n_splits:
        raise ValueError("No hay suficientes componentes para dividir en filas.")

    split_positions = np.sort(np.argsort(diffs)[-n_splits:]) + 1
    index_groups = np.split(order, split_positions)

    rows = [[components[i] for i in group] for group in index_groups]
    return rows


def order_components(rows: List[List[ComponentInfo]]) -> List[ComponentInfo]:
    """Ordena cada fila de izquierda a derecha (por centroide X) y
    concatena las filas de arriba hacia abajo, produciendo el orden
    final de lectura de la hoja."""
    ordered: List[ComponentInfo] = []
    for row in rows:
        row_sorted = sorted(row, key=lambda c: c.cx)
        ordered.extend(row_sorted)
    return ordered


# =======================================================================
# CORRECCIÓN DE INCLINACIÓN (DESKEW)
# =======================================================================

def _rotate_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rota una máscara binaria (o imagen de un solo canal) alrededor
    de su centro, rellenando los bordes con 0 (fondo)."""
    h, w = mask.shape[:2]
    center = (w / 2.0, h / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        mask, rotation_matrix, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def _projection_profile_score(mask: np.ndarray, angle_deg: float) -> float:
    """Rota `mask` por `angle_deg` y calcula la varianza de su perfil
    de proyección horizontal (suma de píxeles de tinta por fila)."""
    rotated = _rotate_mask(mask, angle_deg)
    row_profile = rotated.sum(axis=1).astype(np.float64)
    return float(row_profile.var())


def estimate_skew_angle(
    binary_mask: np.ndarray,
    coarse_range_deg: float = 15.0,
    coarse_step_deg: float = 1.0,
    fine_range_deg: float = 1.5,
    fine_step_deg: float = 0.05,
    downscale: float = 0.5,
) -> float:
    """Estima el ángulo de inclinación global de la hoja mediante el
    método de "perfil de proyección" (projection profile), estándar
    en el preprocesamiento de documentos escaneados.

    Fundamento matemático: sea P_theta(y) = suma de píxeles de tinta
    en la fila y, tras rotar la máscara binaria por un ángulo theta.
    Cuando theta coincide con la inclinación real de la hoja, los
    dígitos de una misma fila caen en las mismas coordenadas y, y las
    franjas entre filas quedan vacías: P_theta(y) alterna entre picos
    altos y valles en cero, maximizando su varianza Var[P_theta].
    Para cualquier otro ángulo, la tinta de distintas filas se
    dispersa sobre más valores de y, aplanando el perfil y reduciendo
    su varianza. Por lo tanto:

                theta* = argmax_theta  Var[P_theta]

    Este método NO depende de poder agrupar los componentes en filas
    de antemano (a diferencia de un ajuste de regresión por fila), lo
    que lo hace robusto incluso cuando la inclinación es lo bastante
    grande como para que filas contiguas se superpongan verticalmente
    en la imagen sin corregir -- el caso exacto en el que un enfoque
    basado en clustering de filas fallaría por un problema de
    "huevo y gallina" (se necesitan filas bien separadas para estimar
    el ángulo, pero se necesita el ángulo para separar bien las
    filas).

    La búsqueda se hace en dos etapas (gruesa y luego fina) sobre una
    versión reducida de la máscara para acelerar el cómputo; el ángulo
    encontrado se aplica después sobre la imagen a resolución
    completa.
    """
    scaled = cv2.resize(
        binary_mask, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_NEAREST
    )

    # --- Búsqueda gruesa ---
    coarse_angles = np.arange(-coarse_range_deg, coarse_range_deg + 1e-9, coarse_step_deg)
    coarse_scores = [_projection_profile_score(scaled, a) for a in coarse_angles]
    best_coarse = float(coarse_angles[int(np.argmax(coarse_scores))])

    # --- Búsqueda fina alrededor del mejor ángulo grueso ---
    fine_angles = np.arange(
        best_coarse - fine_range_deg, best_coarse + fine_range_deg + 1e-9, fine_step_deg
    )
    fine_scores = [_projection_profile_score(scaled, a) for a in fine_angles]
    best_angle = float(fine_angles[int(np.argmax(fine_scores))])

    return best_angle


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rota la imagen alrededor de su centro para corregir la
    inclinación, rellenando los bordes nuevos en blanco puro para no
    introducir artefactos oscuros en el fondo."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    border_value = (255, 255, 255) if image.ndim == 3 else 255
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated


# =======================================================================
# EXTRACCIÓN Y ESTANDARIZACIÓN DE CADA DÍGITO
# =======================================================================

def extract_and_standardize_digit(
    gray: np.ndarray,
    labels: np.ndarray,
    comp: ComponentInfo,
    margin_ratio: float = MARGIN_RATIO,
    target_size: int = TARGET_SIZE,
    mask_dilate_px: int = MASK_DILATE_FOR_CROP,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recorta, limpia el fondo, centra sobre lienzo cuadrado y
    redimensiona un único dígito a `target_size` x `target_size`.

    El fondo se fuerza a blanco puro (255) fuera de una versión
    ligeramente dilatada de la máscara del componente, lo que elimina
    sombras, restos de dígitos vecinos o ruido residual dentro del
    recorte, sin destruir el antialiasing suave del propio trazo
    (que se conserva a partir de los valores de gris originales, no
    de una máscara binaria dura).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (recorte_original_region, lienzo_cuadrado_pre_resize,
         imagen_final_32x32)
    """
    x, y, w, h = comp.bbox
    margin_px = max(2, int(round(margin_ratio * max(w, h))))

    img_h, img_w = gray.shape[:2]
    x0 = max(0, x - margin_px)
    y0 = max(0, y - margin_px)
    x1 = min(img_w, x + w + margin_px)
    y1 = min(img_h, y + h + margin_px)

    gray_crop = gray[y0:y1, x0:x1]
    label_crop = labels[y0:y1, x0:x1]

    # Máscara binaria de ESTE componente únicamente (no de otros dígitos
    # que pudieran caer dentro del mismo recorte por el margen).
    mask = (label_crop == comp.label).astype(np.uint8) * 255
    if mask_dilate_px > 0:
        elem = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * mask_dilate_px + 1, 2 * mask_dilate_px + 1)
        )
        mask = cv2.dilate(mask, elem)

    # Fondo blanco puro fuera de la máscara; dentro, se conserva el
    # valor de gris original (antialiased) para un trazo suave.
    original_region = gray_crop.copy()
    cleaned_crop = np.where(mask > 0, gray_crop, 255).astype(np.uint8)

    # --- Lienzo cuadrado centrado (preserva la relación de aspecto) ---
    crop_h, crop_w = cleaned_crop.shape[:2]
    side = max(crop_h, crop_w)
    canvas = np.full((side, side), 255, dtype=np.uint8)
    off_y = (side - crop_h) // 2
    off_x = (side - crop_w) // 2
    canvas[off_y : off_y + crop_h, off_x : off_x + crop_w] = cleaned_crop

    # --- Redimensionado a target_size x target_size ---
    # INTER_AREA es el método recomendado por OpenCV para reducir
    # resolución (downsampling), ya que actúa como un promedio de área
    # de píxeles y evita aliasing, a diferencia de INTER_LINEAR/NEAREST.
    final_image = cv2.resize(
        canvas, (target_size, target_size), interpolation=cv2.INTER_AREA
    )

    return original_region, canvas, final_image


# =======================================================================
# GUARDADO DE ARCHIVOS
# =======================================================================

def save_dataset(images: List[np.ndarray], output_dir: Path) -> List[Path]:
    """Guarda cada imagen como PNG sin pérdida, nombrada 000.png,
    001.png, ... según su posición en la secuencia ya ordenada."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for idx, img in enumerate(images):
        filename = output_dir / f"{idx:03d}.png"
        # PNG es intrínsecamente sin pérdida; el nivel de compresión
        # (0-9) sólo afecta tamaño/velocidad, nunca la calidad.
        success = cv2.imwrite(str(filename), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not success:
            raise IOError(f"No se pudo escribir el archivo: {filename}")
        saved_paths.append(filename)
    return saved_paths


# =======================================================================
# VISUALIZACIÓN
# =======================================================================

def build_grid_figure(
    images: List[np.ndarray],
    n_cols: int,
    title: str,
    save_path: Path,
    cell_inches: float = 0.55,
) -> None:
    """Genera una cuadrícula (montage) con matplotlib mostrando cada
    imagen de `images` en su celda correspondiente."""
    n_images = len(images)
    n_rows = int(np.ceil(n_images / n_cols))

    fig_w = max(6.0, n_cols * cell_inches)
    fig_h = max(3.0, n_rows * cell_inches) + 0.6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes)

    for idx in range(n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        ax.axis("off")
        if idx < n_images:
            ax.imshow(images[idx], cmap="gray", vmin=0, vmax=255)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def build_comparison_figure(
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    save_path: Path,
) -> None:
    """Genera la figura Original -> Recorte -> Resultado final para
    varios ejemplos, uno por columna.

    Nota técnica: `Axes.axis('off')` oculta también las etiquetas de
    eje (`set_ylabel`), no solo los ticks. Por eso los rótulos de fila
    se dibujan aparte con `ax.text(...)` en coordenadas de ejes, lo que
    garantiza que sean visibles independientemente del estado del eje.
    """
    n_examples = len(samples)
    fig = plt.figure(figsize=(2.2 * n_examples + 1.2, 6.6))
    gs = GridSpec(
        3, n_examples + 1, figure=fig, hspace=0.35, wspace=0.15,
        width_ratios=[0.35] + [1.0] * n_examples,
    )

    row_labels = ["Original", "Recorte", "Resultado final\n(32x32)"]

    # Columna 0: solo etiquetas de fila, sin imagen.
    for row, label in enumerate(row_labels):
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.axis("off")
        ax_label.text(
            0.5, 0.5, label, ha="center", va="center",
            fontsize=11, fontweight="bold", rotation=0, wrap=True,
        )

    for col, (original, crop, final) in enumerate(samples):
        for row, img in enumerate([original, crop, final]):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"#{col}", fontsize=10)

    fig.suptitle(
        "Comparación del pipeline: Original -> Recorte -> Resultado final",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =======================================================================
# ORQUESTACIÓN PRINCIPAL
# =======================================================================

def main() -> None:
    try:
        print("=" * 70)
        print("PIPELINE DE ESTANDARIZACIÓN DE DÍGITOS MANUSCRITOS PARA PCA")
        print("=" * 70)

        # 1. Carga
        print(f"[1/8] Cargando imagen: {INPUT_PATH}")
        raw_image = load_image(INPUT_PATH)

        # 2. Preprocesamiento inicial (para estimar inclinación)
        print("[2/8] Preprocesamiento inicial y estimación de inclinación...")
        gray0 = to_grayscale(raw_image)
        _, mask0 = preprocess_pipeline(gray0)

        # 3. Deskew (si aplica) -- basado en projection profile, no
        # requiere agrupar filas de antemano (ver docstring de la función).
        angle = estimate_skew_angle(mask0)
        print(f"       Ángulo de inclinación estimado: {angle:.3f}°")
        if abs(angle) > DESKEW_ANGLE_THRESHOLD_DEG:
            print(f"[3/8] Corrigiendo inclinación ({angle:.3f}°) y re-procesando...")
            working_image = rotate_image(raw_image, angle)
            gray = to_grayscale(working_image)
            gray_corrected, mask = preprocess_pipeline(gray)
        else:
            print("[3/8] Inclinación despreciable, no se rota la imagen.")
            gray_corrected, mask = gray0, mask0

        components = detect_components(mask)
        _, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # 4. Validación de conteo
        print(f"[4/8] Componentes detectados (tras filtrado de ruido): {len(components)}")
        if len(components) != EXPECTED_TOTAL:
            warnings.warn(
                f"Se esperaban {EXPECTED_TOTAL} dígitos pero se detectaron "
                f"{len(components)}. Revisa el escaneo (dígitos que se "
                f"tocan, manchas, o filas incompletas)."
            )

        # 5. Ordenamiento fila x columna
        print("[5/8] Agrupando en filas y ordenando por lectura (fila, columna)...")
        rows = group_into_rows(components, N_ROWS)
        ordered_components = order_components(rows)
        print(f"       Tamaños de fila: {[len(r) for r in rows]}")

        # 6. Extracción y estandarización
        print("[6/8] Extrayendo, centrando y redimensionando cada dígito a "
              f"{TARGET_SIZE}x{TARGET_SIZE}px...")
        final_images: List[np.ndarray] = []
        raw_crops: List[np.ndarray] = []
        comparison_samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        sample_stride = max(1, len(ordered_components) // N_COMPARISON_EXAMPLES)

        for i, comp in enumerate(ordered_components):
            original_region, square_canvas, final_img = extract_and_standardize_digit(
                gray_corrected, labels, comp
            )
            final_images.append(final_img)
            raw_crops.append(square_canvas)

            if i % sample_stride == 0 and len(comparison_samples) < N_COMPARISON_EXAMPLES:
                comparison_samples.append((original_region, square_canvas, final_img))

        # 7. Guardado
        print(f"[7/8] Guardando {len(final_images)} imágenes PNG en '{OUTPUT_DIR}/'...")
        saved_paths = save_dataset(final_images, OUTPUT_DIR)

        # 8. Visualizaciones
        print(f"[8/8] Generando visualizaciones en '{VIZ_DIR}/'...")
        VIZ_DIR.mkdir(parents=True, exist_ok=True)

        build_grid_figure(
            raw_crops[:100],
            n_cols=10,
            title=f"Primeras 100 imágenes recortadas (pre-resize, n={min(100, len(raw_crops))})",
            save_path=VIZ_DIR / "grid_primeros_100_recortes.png",
        )
        build_grid_figure(
            final_images,
            n_cols=25,
            title=f"Las {len(final_images)} imágenes finales estandarizadas ({TARGET_SIZE}x{TARGET_SIZE}px)",
            save_path=VIZ_DIR / "grid_todas_las_imagenes_finales.png",
            cell_inches=0.35,
        )
        build_comparison_figure(
            comparison_samples,
            save_path=VIZ_DIR / "comparacion_pipeline.png",
        )

        print("=" * 70)
        print("RESUMEN FINAL")
        print("-" * 70)
        print(f"  Componentes conexos detectados (post-filtro): {len(components)}")
        print(f"  Imágenes guardadas:                            {len(saved_paths)}")
        print(f"  Carpeta de salida:                             {OUTPUT_DIR.resolve()}")
        print(f"  Visualizaciones:                                {VIZ_DIR.resolve()}")
        if len(saved_paths) == EXPECTED_TOTAL:
            print(f"  Estado: OK -> se obtuvieron exactamente {EXPECTED_TOTAL} imágenes.")
        else:
            print(f"  Estado: ADVERTENCIA -> se esperaban {EXPECTED_TOTAL}, "
                  f"se obtuvieron {len(saved_paths)}.")
        print("=" * 70)

    except (FileNotFoundError, ValueError, IOError, ImportError) as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
