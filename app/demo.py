"""Gradio demo for the full water meter reading pipeline.

Shows all pipeline stages visually: detection -> crop -> hybrid orientation
(red-position cue / dual-OCR fallback) -> masking -> OCR reading.

Usage:
    python app/demo.py
    python app/demo.py --share
    python app/demo.py --port 7861
"""

import argparse
import sys
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
    ) -> tuple[str, str, list]:
        """Run pipeline and return results for Gradio outputs."""
        if image is None:
            return "", "No image provided.", []

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = pipeline.predict(
            bgr,
            validation_threshold=validation_threshold,
            collect_intermediates=True,
        )

        reading_text = result.reading or ""

        lines = [
            f"Status:          {result.status}",
            f"Reading:         {result.reading or '(none)'}",
            f"Global conf:     {result.global_confidence:.1%}",
            f"  Detection:     {result.detection_confidence:.1%}",
            f"  Recognition:   {result.recognition_confidence:.1%}",
            f"  Orientation:   {result.orientation_confidence:.1%} ({result.orientation_method})",
            "",
            f"Time total:      {result.timing_ms.get('total', 0):.0f} ms",
            f"  Detection:     {result.timing_ms.get('detection', 0):.0f} ms",
            f"  Preprocessing: {result.timing_ms.get('preprocessing', 0):.0f} ms",
            f"  OCR:           {result.timing_ms.get('ocr', 0):.0f} ms",
            f"    OCR passes:  {result.timing_ms.get('ocr_passes', 0)}",
            f"    Orientation: {result.timing_ms.get('orientation_ocr', 0):.0f} ms",
        ]
        if result.warnings:
            lines.append("")
            for warning in result.warnings:
                lines.append(f"Warning: {warning}")

        info_text = "\n".join(lines)

        gallery = []
        stage_labels = [
            ("annotated", "1. Detection"),
            ("crop", "2. OBB Crop"),
            ("oriented", "3. Oriented"),
            ("masked", "4. Masked"),
        ]
        for key, label in stage_labels:
            img = result.intermediates.get(key)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gallery.append((rgb, label))

        return reading_text, info_text, gallery

    def clear_all() -> tuple[None, str, str, list]:
        """Reset input and outputs to the initial empty state."""
        return None, "", "", []

    with gr.Blocks(title="Water Meter Reading Demo") as demo:
        gr.Markdown(
            """
        # Water Meter Reading Demo

        Upload an image of a water meter to automatically read the odometer.
        The pipeline uses **YOLO OBB** for detection and **PaddleOCR** (fine-tuned) for digit recognition.

        **Detector:** YOLOv8n-OBB (baseline) | **OCR:** en_PP-OCRv4_mobile_rec (fine-tuned, digit-only)
        """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=0, min_width=180):
                read_btn = gr.Button("Read Meter", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", variant="secondary")
            with gr.Column(scale=1, min_width=320):
                conf_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    label="Validation Threshold",
                    info="Minimum global confidence required to mark a reading as valid.",
                )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    type="numpy",
                    sources=["upload", "webcam", "clipboard"],
                )

            with gr.Column(scale=1):
                reading_output = gr.Textbox(
                    label="Reading",
                    lines=1,
                    max_lines=1,
                )
                info_output = gr.Textbox(
                    label="Details",
                    lines=10,
                )
                with gr.Accordion("Pipeline Stages", open=False):
                    stage_gallery = gr.Gallery(
                        label="Pipeline Stages",
                        columns=2,
                        object_fit="contain",
                        height=280,
                    )

        outputs = [reading_output, info_output, stage_gallery]

        read_btn.click(
            fn=process_image,
            inputs=[input_image, conf_slider],
            outputs=outputs,
        )
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[input_image, reading_output, info_output, stage_gallery],
        )

        gr.Markdown(
            """
        ---
        **Pipeline stages shown in gallery:**
        1. **Detection** - YOLO OBB overlay on original image
        2. **OBB Crop** - extracted odometer region (aspect-ratio corrected)
        3. **Oriented** - hybrid 0°/180° correction (red-position cue → dual-OCR fallback)
        4. **Masked** - after red decimal digit removal (LAB v4)

        The OCR reader then processes the masked crop to produce the final reading.
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
