"""Test ONNX model inference to understand output format."""
import onnxruntime as ort
import cv2
import numpy as np

# Load session
session = ort.InferenceSession('weights/field_detector.onnx', providers=['CPUExecutionProvider'])

# Create a test image (simulate an ID card)
test_img = np.zeros((600, 900, 3), dtype=np.uint8)
test_img[100:200, 100:400] = [200, 200, 200]  # Simulate a field
test_img[300:400, 100:400] = [180, 180, 180]  # Another field

# Preprocess
def preprocess(image, input_size=(640, 640)):
    h, w = image.shape[:2]
    scale = min(input_size[0] / w, input_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    pad_w = (input_size[0] - new_w) // 2
    pad_h = (input_size[1] - new_h) // 2
    canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    batch = np.expand_dims(chw, axis=0)
    
    return batch, (w / new_w, h / new_h), (pad_w, pad_h)

input_tensor, scale, padding = preprocess(test_img)
print(f"Input shape: {input_tensor.shape}")

# Run inference
outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
output = outputs[0]

print(f"\n=== OUTPUT TENSOR ===")
print(f"Shape: {output.shape}")
print(f"Dtype: {output.dtype}")
print(f"Min: {output.min():.4f}, Max: {output.max():.4f}, Mean: {output.mean():.4f}")

# Transpose to [anchors, 21]
output_t = np.transpose(output[0], (1, 0))
print(f"\nTransposed shape: {output_t.shape}")

# Analyze structure
boxes = output_t[:, 0:4]
scores = output_t[:, 4:21]

print(f"\n=== BOXES ===")
print(f"Shape: {boxes.shape}")
print(f"Min: {boxes.min():.4f}, Max: {boxes.max():.4f}")
print(f"Sample box: {boxes[0]}")

print(f"\n=== CLASS SCORES ===")
print(f"Shape: {scores.shape}")
print(f"Min: {scores.min():.4f}, Max: {scores.max():.4f}")

# Find anchors with high scores
max_scores_per_anchor = np.max(scores, axis=1)
print(f"\n=== SCORE DISTRIBUTION ===")
print(f"Max per anchor - Min: {max_scores_per_anchor.min():.4f}, Max: {max_scores_per_anchor.max():.4f}")
print(f"Anchors > 0.5: {np.sum(max_scores_per_anchor > 0.5)}")
print(f"Anchors > 0.3: {np.sum(max_scores_per_anchor > 0.3)}")
print(f"Anchors > 0.2: {np.sum(max_scores_per_anchor > 0.2)}")
print(f"Anchors > 0.1: {np.sum(max_scores_per_anchor > 0.1)}")

# Check per-class max scores
class_names = ['first_name', 'last_name', 'add_line_1', 'add_line_2', 'front_nid', 
               'back_nid', 'serial_num', 'issue_code', 'expiry_date', 'job_title',
               'gender', 'religion', 'marital_status', 'face', 'front_logo', 'address', 'dob']

print(f"\n=== PER-CLASS MAX SCORES ===")
for cls_id in range(17):
    cls_max = np.max(scores[:, cls_id])
    if cls_max > 0.1:
        print(f"{cls_id:2d} ({class_names[cls_id]:15s}): {cls_max:.4f}")

# Find top detections
print(f"\n=== TOP 10 DETECTIONS ===")
top_indices = np.argsort(max_scores_per_anchor)[::-1][:10]
for idx in top_indices:
    best_cls = np.argmax(scores[idx])
    print(f"Anchor {idx}: box={boxes[idx]}, class={best_cls} ({class_names[best_cls]}), score={scores[idx, best_cls]:.4f}")
