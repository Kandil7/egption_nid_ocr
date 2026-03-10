"""Inspect YOLO model class names."""
from ultralytics import YOLO

# Load the fields model
model = YOLO('weights/detect_odjects.pt')

print("=== detect_odjects.pt ===")
print(f"Model type: {type(model)}")
print(f"Class names: {model.names}")

# Also check card detection model
try:
    card_model = YOLO('weights/detect_id_card.pt')
    print("\n=== detect_id_card.pt ===")
    print(f"Class names: {card_model.names}")
except Exception as e:
    print(f"\nCard model error: {e}")

# Also check NID model
try:
    nid_model = YOLO('weights/detect_id.pt')
    print("\n=== detect_id.pt ===")
    print(f"Class names: {nid_model.names}")
except Exception as e:
    print(f"\nNID model error: {e}")
