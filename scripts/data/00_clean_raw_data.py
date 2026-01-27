"""
Script para limpiar datos raw del dataset de medidores de agua.

Soporta múltiples lotes organizados en carpetas:
- Formato nuevo: batch_01, batch_02, batch_03, etc.
- Formato legacy: 01, 02, 03, etc.

Cada lote contiene readings.csv y subcarpetas con fotos.

Input: data/raw/batch_01/readings.csv, data/raw/batch_01/fotos1/, etc.
       data/raw/batch_02/readings.csv, data/raw/batch_02/fotos1/, etc.
Output: data/cleaned/images/ (secuencial, comprimidas a 1280px max, JPEG Q90)
        data/cleaned/metadata.csv (incluye batch_id para trazabilidad)

Uso:
    python 0_clean_raw_data.py [--max-dim 1280] [--jpeg-quality 90] [--workers 4]
"""

import argparse
import pandas as pd
import shutil
import sys
from pathlib import Path
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ExifTags
from tqdm import tqdm

# Base directory (repo root)
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
from utils.logging import setup_logger

logger = setup_logger("00_clean_data")

# Rutas
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
IMAGES_DIR = CLEANED_DIR / "images"


# ─────────────────────────────────────────────────────────────────────────────
# Image processing (from 1_prepare_for_annotation.py)
# ─────────────────────────────────────────────────────────────────────────────

def apply_exif_orientation(image: Image.Image) -> Image.Image:
    """Aplica la rotación correcta basándose en los metadatos EXIF."""
    try:
        exif = image._getexif()
        if exif is not None:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == 'Orientation':
                    orientation = value
                    break
            else:
                return image
        else:
            return image
    except (AttributeError, KeyError, IndexError):
        return image

    if orientation == 2:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 3:
        image = image.rotate(180)
    elif orientation == 4:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    elif orientation == 5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
    elif orientation == 6:
        image = image.rotate(-90, expand=True)
    elif orientation == 7:
        image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(-90, expand=True)
    elif orientation == 8:
        image = image.rotate(90, expand=True)

    return image


def resize_image(image: Image.Image, max_dim: int) -> Image.Image:
    """Redimensiona la imagen manteniendo la relación de aspecto."""
    width, height = image.size
    if max(width, height) <= max_dim:
        return image

    if width > height:
        new_width = max_dim
        new_height = int(height * (max_dim / width))
    else:
        new_height = max_dim
        new_width = int(width * (max_dim / height))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def compress_image(source_path: Path, dest_path: Path, max_dim: int, jpeg_quality: int):
    """
    Procesa una imagen: corrige EXIF, redimensiona y comprime.

    Returns:
        True si se procesó correctamente, False si hubo error.
    """
    try:
        with Image.open(source_path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            img = apply_exif_orientation(img)
            img = resize_image(img, max_dim)

            img.save(dest_path, 'JPEG', quality=jpeg_quality, optimize=True)
            return True
    except Exception as e:
        logger.error(f"Error procesando {source_path.name}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Limpia y comprime imágenes raw.")
    parser.add_argument("--max-dim", type=int, default=1280,
                        help="Dimensión máxima del lado más largo (default: 1280)")
    parser.add_argument("--jpeg-quality", type=int, default=90,
                        help="Calidad JPEG 1-100 (default: 90)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Threads para procesamiento paralelo (default: 4)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("LIMPIEZA Y COMPRESIÓN DE DATOS RAW")
    logger.info(f"  max_dim={args.max_dim}, jpeg_quality={args.jpeg_quality}, workers={args.workers}")
    logger.info("=" * 60)

    # Verificar si existe la carpeta cleaned
    if CLEANED_DIR.exists():
        response = input(
            f"La carpeta '{CLEANED_DIR}' ya existe. ¿Desea eliminar su contenido y volver a crearla? (s/N): "
        ).strip().lower()
        if response in ('s', 'si'):
            logger.info(f"Eliminando contenido de {CLEANED_DIR}")
            shutil.rmtree(CLEANED_DIR)
        else:
            logger.info("Operación cancelada por el usuario.")
            return

    # Crear directorios
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorio creado: {IMAGES_DIR}")

    # 1. Descubrir todos los lotes en data/raw/
    logger.info("Descubriendo lotes...")
    batch_dirs = sorted([
        d for d in RAW_DIR.iterdir()
        if d.is_dir() and (d.name.startswith("batch_") or d.name.isdigit())
    ])

    if not batch_dirs:
        logger.error(f"No se encontraron lotes en {RAW_DIR}")
        return

    logger.info(f"Lotes encontrados: {[d.name for d in batch_dirs]}")

    # 2. Leer y consolidar datos de todos los lotes
    all_data = []
    batch_stats = {}

    for batch_dir in batch_dirs:
        batch_id = batch_dir.name.replace("batch_", "") if batch_dir.name.startswith("batch_") else batch_dir.name
        readings_file = batch_dir / "readings.csv"

        if not readings_file.exists():
            logger.warning(f"No se encontró readings.csv en lote {batch_id}, omitiendo...")
            continue

        logger.info(f"Procesando lote {batch_id}...")
        df_batch = pd.read_csv(readings_file)

        df_batch = df_batch.dropna(subset=['SERIAL', 'LECTURA'])
        df_batch = df_batch[df_batch['SERIAL'].astype(str).str.isnumeric()]

        df_batch['batch_id'] = int(batch_id)
        df_batch['CARPETA DESTINO'] = batch_dir.name + '/' + df_batch['CARPETA DESTINO'].str.rstrip('/')

        all_data.append(df_batch)
        batch_stats[batch_id] = len(df_batch)
        logger.info(f"  Lote {batch_id}: {len(df_batch)} registros válidos")

    # Consolidar todos los datos
    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"\nTotal registros consolidados: {len(df)}")

    # 3. Crear DataFrame con IDs secuenciales globales
    logger.info("\nGenerando IDs secuenciales globales...")
    temp_df = pd.DataFrame({
        'id': range(1, len(df) + 1),
        'batch_id': df['batch_id'].astype(int),
        'serial_original': df['SERIAL'].astype(int),
        'reading_value': df['LECTURA'].astype(int),
        'image_filename': [f"{i:05d}.jpg" for i in range(1, len(df) + 1)],
        'source_folder': df['CARPETA DESTINO'].values,
        'source_filepath': ''
    })

    logger.info(f"Registros a procesar: {len(temp_df)}")

    # 4. Localizar imágenes fuente
    logger.info("\nLocalizando imágenes fuente...")
    tasks = []  # (source_path, dest_path, row_idx)
    missing = []

    for idx, row in temp_df.iterrows():
        serial = row['serial_original']
        source_folder = row['source_folder']
        new_filename = row['image_filename']

        source_pattern = RAW_DIR / source_folder / f"*{serial}*.jpg"
        source_files = glob(str(source_pattern))

        if source_files:
            source = Path(source_files[0])
            destination = IMAGES_DIR / new_filename
            source_filepath = source.relative_to(RAW_DIR)
            temp_df.at[idx, 'source_filepath'] = str(source_filepath).replace('\\', '/')
            tasks.append((source, destination, idx))
        else:
            logger.warning(f"  No encontrada: serial {serial} en {source_folder}")
            missing.append(row['id'])

    logger.info(f"Imágenes localizadas: {len(tasks)}/{len(temp_df)}")
    if missing:
        logger.warning(f"Imágenes faltantes: {len(missing)}")

    # 5. Procesar imágenes en paralelo (EXIF + resize + JPEG compression)
    logger.info(f"\nComprimiendo imágenes (max_dim={args.max_dim}, Q={args.jpeg_quality})...")
    copied = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(compress_image, src, dst, args.max_dim, args.jpeg_quality): src.name
            for src, dst, _ in tasks
        }
        with tqdm(total=len(futures), desc="Procesando imágenes") as pbar:
            for future in as_completed(futures):
                if future.result():
                    copied += 1
                else:
                    errors += 1
                    fname = futures[future]
                    # Find and mark as missing
                    for src, dst, row_idx in tasks:
                        if src.name == fname:
                            missing.append(temp_df.at[row_idx, 'id'])
                            break
                pbar.update(1)

    logger.info(f"\nImágenes procesadas: {copied}/{len(tasks)}")
    if errors:
        logger.warning(f"Errores de procesamiento: {errors}")

    # 6. Filtrar: solo imágenes que se procesaron exitosamente
    if missing:
        temp_df = temp_df[~temp_df['id'].isin(missing)]

    # Crear metadata (sin readings.csv separado — metadata.csv es la fuente de verdad)
    metadata_df = temp_df[['id', 'batch_id', 'serial_original', 'reading_value',
                            'image_filename', 'source_filepath']].copy()

    metadata_csv = CLEANED_DIR / "metadata.csv"
    metadata_df.to_csv(metadata_csv, index=False)
    logger.info(f"\nCSV metadata guardado: {metadata_csv}")
    logger.info(f"  Columnas: {list(metadata_df.columns)}")
    logger.info(f"  Registros: {len(metadata_df)}")

    # 7. Resumen
    original_size = sum(src.stat().st_size for src, _, _ in tasks) / (1024 * 1024)
    new_files = list(IMAGES_DIR.glob("*.jpg"))
    new_size = sum(f.stat().st_size for f in new_files) / (1024 * 1024) if new_files else 0

    logger.info("=" * 60)
    logger.info("RESUMEN DE LIMPIEZA")
    logger.info("=" * 60)
    logger.info(f"Lotes procesados: {len(batch_stats)}")
    for bid, count in batch_stats.items():
        logger.info(f"  - Lote {bid}: {count} registros")
    logger.info(f"Total registros finales: {len(metadata_df)}")
    logger.info(f"Imágenes procesadas: {copied}")
    logger.info(f"Tamaño original: {original_size:.1f} MB")
    logger.info(f"Tamaño comprimido: {new_size:.1f} MB")
    if original_size > 0:
        logger.info(f"Reducción: {((1 - new_size / original_size) * 100):.1f}%")
    logger.info(f"Directorio imágenes: {IMAGES_DIR}")
    logger.info(f"CSV metadata: {metadata_csv}")
    logger.info("=" * 60)
    logger.info("Limpieza completada exitosamente!")


if __name__ == "__main__":
    main()
