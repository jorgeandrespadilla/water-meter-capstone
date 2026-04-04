"""Gradio demo for the full water meter reading pipeline.

Shows all pipeline stages visually: detection -> crop -> hybrid orientation
(red-position cue / dual-OCR fallback) -> masking -> OCR reading.

Usage:
    python app/demo.py
    python app/demo.py --share
    python app/demo.py --port 7861
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline import DEFAULT_OCR_MODEL_DIR, DEFAULT_YOLO_PATH, WaterMeterPipeline


def create_demo(
    yolo_model_path: Path = DEFAULT_YOLO_PATH,
    ocr_model_dir: Path = DEFAULT_OCR_MODEL_DIR,
) -> gr.Blocks:
    """Build the Gradio Blocks interface."""
    pipeline = WaterMeterPipeline(
        yolo_model_path=yolo_model_path,
        ocr_model_dir=ocr_model_dir,
    )

    def process_image(
        image: np.ndarray | None,
        validation_threshold: float,
    ) -> dict:
        """Run pipeline and return results for Gradio outputs."""
        if image is None:
            return {
                reading_output: "",
                status_output: "",
                confidence_output: "",
                total_time_output: "",
                warnings_output: "",
                stage_gallery: [],
                debug_output: "",
            }

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = pipeline.predict(
            bgr,
            validation_threshold=validation_threshold,
            collect_intermediates=True,
        )

        reading_text = result.reading or "Sin lectura"

        status_labels = {
            "valid": "Válida",
            "needs_review": "Requiere revisión",
            "no_detection": "Sin detección",
        }
        status_display = status_labels.get(result.status, result.status)
        status_text = status_display
        warnings_text = "\n".join(f"Aviso: {warning}" for warning in result.warnings)
        confidence_text = "\n".join(
            [
                f"Global: {result.global_confidence:.1%}",
                f"Detección: {result.detection_confidence:.1%}",
                f"Reconocimiento: {result.recognition_confidence:.1%}",
            ]
        )
        total_time_text = f"{result.timing_ms.get('total', 0):.0f} ms"

        gallery = []
        stage_labels = [
            ("annotated", "1. Detección"),
            ("crop", "2. Recorte"),
            ("oriented", "3. Orientación"),
            ("masked", "4. Máscara"),
        ]
        for key, label in stage_labels:
            img = result.intermediates.get(key)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gallery.append((rgb, label))

        # Debug: full result as JSON (exclude intermediates — not serializable)
        debug_dict = asdict(result)
        debug_dict.pop("intermediates", None)
        debug_json = json.dumps(debug_dict, indent=2, ensure_ascii=False)

        return {
            reading_output: reading_text,
            status_output: status_text,
            confidence_output: confidence_text,
            total_time_output: total_time_text,
            warnings_output: warnings_text,
            stage_gallery: gallery,
            debug_output: debug_json,
        }

    with gr.Blocks(title="Lectura de Medidores de Agua") as demo:
        gr.Markdown(
            """
        # Lectura Automática de Medidores de Agua

        Sistema para la validación automática de lecturas de medidores de agua potable,
        basado en visión por computador y aprendizaje profundo.
        Analiza la fotografía del medidor, extrae el valor del odómetro y genera
        indicadores de confianza para priorizar la revisión humana.

        Suba una imagen del medidor para obtener la lectura.
        """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=0, min_width=180):
                read_btn = gr.Button("Obtener Lectura", variant="primary", size="lg")
                clear_btn = gr.ClearButton(value="Limpiar", variant="secondary")
            with gr.Column(scale=1, min_width=320):
                conf_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    label="Umbral de validación",
                    info="Confianza mínima para considerar una lectura como válida.",
                )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Imagen de entrada",
                    type="numpy",
                    sources=["upload", "webcam", "clipboard"],
                )

            with gr.Column(scale=1):
                reading_output = gr.Textbox(
                    label="Lectura",
                    lines=1,
                    max_lines=1,
                )
                status_output = gr.Textbox(
                    label="Estado",
                    lines=1,
                    max_lines=1,
                )
                confidence_output = gr.Textbox(
                    label="Confianza",
                    lines=3,
                )
                total_time_output = gr.Textbox(
                    label="Tiempo total",
                    lines=1,
                    max_lines=1,
                )
                warnings_output = gr.Textbox(
                    label="Advertencias",
                    lines=3,
                )
                stage_gallery = gr.Gallery(
                    label="Etapas del proceso",
                    columns=2,
                    object_fit="contain",
                    height=280,
                )
                with gr.Accordion("Respuesta completa (debug)", open=False):
                    debug_output = gr.Code(
                        label="JSON",
                        language="json",
                        lines=15,
                    )

        outputs = [
            reading_output,
            status_output,
            confidence_output,
            total_time_output,
            warnings_output,
            stage_gallery,
            debug_output,
        ]

        clear_btn.add(
            [
                input_image,
                reading_output,
                status_output,
                confidence_output,
                total_time_output,
                warnings_output,
                stage_gallery,
                debug_output,
            ]
        )

        read_btn.click(
            fn=process_image,
            inputs=[input_image, conf_slider],
            outputs=outputs,
        )

        gr.Markdown(
            """
        ---
        **Etapas del proceso:**
        1. **Detección** — localización del odómetro en la imagen
        2. **Recorte** — extracción de la región del odómetro
        3. **Orientación** — corrección automática de rotación (0° / 180°)
        4. **Máscara** — eliminación de dígitos decimales (rojos)

        Finalmente, el modelo OCR procesa la imagen preparada para obtener la lectura.
        """
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Water Meter Reading Demo")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_YOLO_PATH,
        help="Path to YOLO model file",
    )
    parser.add_argument(
        "--ocr-model-dir",
        type=Path,
        default=DEFAULT_OCR_MODEL_DIR,
        help="Path to fine-tuned OCR model directory",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on",
    )
    args = parser.parse_args()

    demo = create_demo(
        yolo_model_path=args.model,
        ocr_model_dir=args.ocr_model_dir,
    )

    print(f"Starting Gradio server on port {args.port}...", flush=True)
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
