"""
Script to export YOLO models to ONNX format for CPU inference.
Run this to convert trained YOLO models to ONNX.
"""

import os
import sys


def export_yolo_to_onnx(model_path: str, output_path: str, img_size: int = 640):
    """
    Export YOLO model to ONNX format.

    Args:
        model_path: Path to YOLO .pt model
        output_path: Path for ONNX output
        img_size: Input image size
    """
    try:
        from ultralytics import YOLO

        print(f"Loading model: {model_path}")
        model = YOLO(model_path)

        print(f"Exporting to ONNX (img_size={img_size})...")
        model.export(
            format="onnx", imgsz=img_size, optimize=True, opset=12, simplify=True
        )

        # Rename if needed
        base = model_path.replace(".pt", "")
        onnx_file = f"{base}.onnx"

        if os.path.exists(onnx_file) and onnx_file != output_path:
            os.rename(onnx_file, output_path)
            print(f"Saved to: {output_path}")

        print(f"Export complete!")
        return True

    except Exception as e:
        print(f"Error exporting model: {e}")
        return False


def main():
    """Export YOLO models to ONNX format."""
    print("=" * 50)
    print("Export YOLO Models to ONNX")
    print("=" * 50)

    # Define models to export
    models = [
        ("weights/yolo_card_detect.pt", "weights/yolo_card_detect.onnx"),
        ("weights/yolo_fields_detect.pt", "weights/yolo_fields_detect.onnx"),
    ]

    for model_path, output_path in models:
        if not os.path.exists(model_path):
            print(f"\nModel not found: {model_path}")
            print(
                "Please ensure you have the trained YOLO models in the weights/ directory"
            )
            print("\nTo get models:")
            print("  1. Clone: git clone https://github.com/NASO7Y/ocr_egyptian_ID.git")
            print("  2. Copy: cp OCR_Egyptian_ID/weights/*.pt weights/")
            continue

        print(f"\n--- Exporting {model_path} ---")
        success = export_yolo_to_onnx(model_path, output_path)

        if success:
            print(f"Exported: {output_path}")

    print("\n" + "=" * 50)
    print("Export complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
