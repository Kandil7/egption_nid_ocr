"""Test ONNX model with real image."""
import onnxruntime as ort
import cv2
import numpy as np

# Load session
session = ort.InferenceSession('weights/field_detector.onnx', providers=['CPUExecutionProvider'])

# Load real image
img_path = 'tests/sample_ids/images/img_10_png.rf.d68824143abe697075c21dbd67223ef7.jpg'
image = cv2.imread(img_path)
print(f"Loaded image: {image.shape}")

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

input_tensor, scale, padding = preprocess(image)

# Run inference
outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
output = outputs[0]

# Transpose to [anchors, 21]
output_t = np.transpose(output[0], (1, 0))

# Analyze
boxes = output_t[:, 0:4]
scores = output_t[:, 4:21]

class_names = ['first_name', 'last_name', 'add_line_1', 'add_line_2', 'front_nid', 
               'back_nid', 'serial_num', 'issue_code', 'expiry_date', 'job_title',
               'gender', 'religion', 'marital_status', 'face', 'front_logo', 'address', 'dob']

# Find anchors with high scores
max_scores_per_anchor = np.max(scores, axis=1)

print(f"\n=== SCORE DISTRIBUTION ===")
print(f"Max per anchor - Min: {max_scores_per_anchor.min():.4f}, Max: {max_scores_per_anchor.max():.4f}")
print(f"Anchors > 0.5: {np.sum(max_scores_per_anchor > 0.5)}")
print(f"Anchors > 0.3: {np.sum(max_scores_per_anchor > 0.3)}")
print(f"Anchors > 0.18: {np.sum(max_scores_per_anchor > 0.18)}")
print(f"Anchors > 0.1: {np.sum(max_scores_per_anchor > 0.1)}")

# Check per-class max scores
print(f"\n=== PER-CLASS MAX SCORES ===")
for cls_id in range(17):
    cls_max = np.max(scores[:, cls_id])
    if cls_max > 0.15:
        print(f"{cls_id:2d} ({class_names[cls_id]:15s}): {cls_max:.4f}")

# Find top detections
print(f"\n=== TOP 20 DETECTIONS (score > 0.15) ===")
valid_indices = np.where(max_scores_per_anchor > 0.15)[0]
print(f"Found {len(valid_indices)} anchors with score > 0.15")

# Sort by score
sorted_indices = np.argsort(max_scores_per_anchor)[::-1][:20]
for idx in sorted_indices:
    score = max_scores_per_anchor[idx]
    if score > 0.15:
        best_cls = np.argmax(scores[idx])
        cx, cy, w, h = boxes[idx]
        print(f"Score={score:.4f}, class={best_cls} ({class_names[best_cls]:15s}), box=[{cx:.1f}, {cy:.1f}, {w:.1f}, {h:.1f}]")
