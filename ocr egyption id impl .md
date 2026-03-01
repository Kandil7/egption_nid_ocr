<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# عاوز ابني مشروع ocr egyption id

خلينا نبني الـ roadmap الكاملة للمشروع من الصفر، مع كل المكونات والأدوات المطلوبة.

***

## معمارية المشروع

مشروع OCR للبطاقة الشخصية المصرية بيتكون من **3 مراحل أساسية**: Detection → OCR → Data Parsing.

```
Input Image
     ↓
[YOLOv8 - Card Detection]  ← crop + align البطاقة
     ↓
[YOLOv8 - Field Detection] ← تحديد مناطق الاسم، العنوان، الرقم...
     ↓
[EasyOCR / PaddleOCR]      ← استخراج النص العربي والإنجليزي
     ↓
[ID Number Parser]          ← decode الرقم القومي
     ↓
Structured JSON Output
```


***

## المرحلة 1 — Dataset \& Annotation

أهم خطوة في المشروع هي الـ dataset.[^1_1]

- **Roboflow Universe** — فيه datasets جاهزة لـ Egyptian ID detection، ابحث عن `egyptian-id`[^1_2]
- **أنشئ dataset بنفسك** باستخدام Roboflow للـ annotation على classes:
    - `id_card`, `name`, `address`, `id_number`, `birth_date`, `gender`, `nationality`
- استخدم **data augmentation**: rotation, brightness, blur (ضروري عشان الصور الواقعية بتتصور بكاميرات مش مثالية)

***

## المرحلة 2 — نموذج الـ Detection

استخدم **YOLOv8** من Ultralytics — الأنسب للمشاريع دي:[^1_3]

```python
from ultralytics import YOLO

# Model 1: detect ID card من الصورة كلها
model_card = YOLO("yolov8n.pt")
model_card.train(data="id_card.yaml", epochs=50, imgsz=640)

# Model 2: detect الـ fields جوه البطاقة
model_fields = YOLO("yolov8n.pt")
model_fields.train(data="id_fields.yaml", epochs=100, imgsz=640)
```

> **نصيحة:** استخدم نموذجين منفصلين — واحد للكارت نفسه، وواحد للحقول جوّاه. ده بيزود الدقة بشكل كبير.[^1_3]

***

## المرحلة 3 — OCR Engine

المقارنة بين الـ engines المتاحة:[^1_4][^1_1]


| Engine | دعم العربي | الدقة | السرعة | الملاحظات |
| :-- | :-- | :-- | :-- | :-- |
| **EasyOCR** | ✅ ممتاز | عالية | متوسطة | الأفضل للعربي |
| **PaddleOCR** | ✅ جيد | عالية جداً | سريع | يحتاج fine-tuning |
| **Tesseract** | ⚠️ محدود | متوسطة | سريع | يحتاج Arabic lang pack |
| **ArabicOCR** | ✅ | متوسطة | سريع | مخصص للعربي |

```python
import easyocr
reader = easyocr.Reader(['ar', 'en'])
results = reader.readtext(cropped_field_image)
```


***

## المرحلة 4 — تحليل الرقم القومي

الرقم القومي المصري (14 رقم) بيحتوي على معلومات مشفرة:[^1_5]

```python
def parse_egyptian_id(national_id: str) -> dict:
    century_code = int(national_id[^1_0])
    year = int(national_id[1:3])
    month = int(national_id[3:5])
    day = int(national_id[5:7])
    gov_code = int(national_id[7:9])
    gender_digit = int(national_id[^1_12])

    century = 1900 if century_code == 2 else 2000
    full_year = century + year

    governorates = {
        1: "القاهرة", 2: "الإسكندرية", 3: "بورسعيد",
        4: "السويس", 11: "دمياط", 12: "الدقهلية", # ... إلخ
    }

    return {
        "birth_date": f"{day}/{month}/{full_year}",
        "governorate": governorates.get(gov_code, "غير معروف"),
        "gender": "ذكر" if gender_digit % 2 != 0 else "أنثى"
    }
```


***

## المرحلة 5 — الـ API Layer (FastAPI)

بما إنك شغال على FastAPI، الـ structure التالية مناسبة جداً:[^1_1]

```python
from fastapi import FastAPI, UploadFile, File
import cv2, numpy as np

app = FastAPI()

@app.post("/extract-id")
async def extract_id(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    
    # 1. Detect card
    card_crop = detect_card(img)
    # 2. Detect fields
    fields = detect_fields(card_crop)
    # 3. OCR each field
    extracted = {field: ocr_field(crop) for field, crop in fields.items()}
    # 4. Parse ID number
    extracted["parsed_id"] = parse_egyptian_id(extracted["id_number"])
    
    return extracted
```


***

## Stack الموصى به

- **Detection:** YOLOv8 (Ultralytics)[^1_3]
- **OCR:** EasyOCR للعربي + Tesseract للأرقام[^1_4][^1_1]
- **Backend:** FastAPI + Python
- **Image Processing:** OpenCV + Pillow
- **Annotation:** Roboflow[^1_2]
- **Interface:** Streamlit للـ demo أو FastAPI للـ production[^1_1]

***

## مشاريع Open Source تقدر تستلهم منها

- **[NASO7Y/OCR_Egyptian_ID](https://github.com/NASO7Y/ocr_egyptian_ID)** — YOLO + EasyOCR + Streamlit + Fraud Detection[^1_1]
- **[Mostafa-Emad77/Egyptian-ID-Extraction](https://github.com/Mostafa-Emad77/Egyptian-ID-Extraction)** — ROI extraction + ArabicOCR + Tesseract[^1_4]
- **[keronagy/Egyptian-ID-Data-Extraction](https://github.com/keronagy/Egyptian-ID-Data-Extraction)** — نموذج تعليمي بسيط[^1_6]
<span style="display:none">[^1_10][^1_11][^1_13][^1_14][^1_15][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: https://github.com/NASO7Y/OCR_Egyptian_ID

[^1_2]: https://universe.roboflow.com/search?q=like%3Afsococ%2Fegyptian-id-seg-r4h5w

[^1_3]: https://fci.stafpu.bu.edu.eg/Artificial Intelligence/7323/cv/Hossam Fakher.pdf

[^1_4]: https://github.com/Mostafa-Emad77/Egyptian-ID-Extraction

[^1_5]: https://github.com/ahmedsaleh020/national-id-extractor

[^1_6]: https://github.com/keronagy/Egyptian-ID-Data-Extraction

[^1_7]: https://www.kaggle.com/code/yahyaahmed1012/egyptian-id-ocr

[^1_8]: https://signme.it/arabic-ocr-with-signme-egyptian-id-reading-2/?lang=en

[^1_9]: https://signme.it/egypt-national-id-api/?lang=en

[^1_10]: https://github.com/M-Bahy/Egyptian-ID-reader

[^1_11]: https://m.sinosecu.com.cn/productDetail/0/47?lang=en_US

[^1_12]: https://www.kaggle.com/code/ahmedabdelghagny/egyptian-car-plates-yolov11-fine-tuned-easyocr

[^1_13]: https://www.kaggle.com/code/engsaiedali/ocr-egyptian-id-card

[^1_14]: https://www.facebook.com/groups/ComputerVisionGroup/posts/3339668449510666/

[^1_15]: https://www.facebook.com/groups/one.million.egyptian.coder/posts/1993281871266243/


---

# عاوز مشروع متكامل باستخدام موديلات جاهز من مشاريع فعليه و لها فاعليه عاليه

المشروع المتكامل هيعتمد على **موديلات جاهزة وحقيقية** من مشاريع فعلية بأعلى دقة متاحة، مجمّعة في pipeline واحد.

***

## الموديلات المختارة وليه

أفضل combination موجود حالياً للـ Egyptian ID:[^2_1][^2_2][^2_3]


| المكوِّن | الموديل المختار | الدقة | المصدر |
| :-- | :-- | :-- | :-- |
| **Card Detection** | YOLOv8 (NASO7Y trained) | ~95%+ | GitHub |
| **Field Detection** | YOLOv8 (NASO7Y trained) | ~95%+ | GitHub |
| **Arabic OCR** | `NAMAA-Space/Qari-OCR` | أعلى دقة للعربي 2025 | HuggingFace |
| **English/Numbers** | EasyOCR `en` | عالي | pip |
| **ID Parser** | Custom Python | — | منطق محلي |

> **Qari-OCR** هو أحدث موديل عربي OCR عالي الدقة (يناير 2026)، مبني على Multimodal LLM ومُدرَّب خصيصاً على Arabic text recognition.[^2_3][^2_4]

***

## هيكل المشروع الكامل

```
egyptian_id_ocr/
├── app.py                  # FastAPI main
├── models/
│   ├── detector.py         # YOLO card + field detection
│   ├── ocr_engine.py       # Qari-OCR + EasyOCR
│   └── id_parser.py        # national ID decoder
├── utils/
│   ├── image_processing.py # OpenCV preprocessing
│   └── postprocessing.py   # text cleaning
├── weights/
│   └── yolo_egyptian_id.pt # من مشروع NASO7Y
├── requirements.txt
└── README.md
```


***

## الكود الكامل

### 1. Installation

```bash
git clone https://github.com/NASO7Y/ocr_egyptian_ID.git
pip install ultralytics easyocr transformers torch pillow fastapi uvicorn python-multipart opencv-python
```


### 2. `models/detector.py` — YOLO Detection

```python
from ultralytics import YOLO
import cv2
import numpy as np

class EgyptianIDDetector:
    def __init__(self, model_path="weights/yolo_egyptian_id.pt"):
        # موديل NASO7Y المدرب على Egyptian ID
        self.model = YOLO(model_path)
        self.field_labels = {
            0: "name", 1: "address", 2: "id_number",
            3: "birth_date", 4: "gender", 5: "nationality"
        }

    def detect_card(self, image: np.ndarray) -> np.ndarray:
        """Crop the ID card from full image"""
        results = self.model(image)[^2_0]
        for box in results.boxes:
            if int(box.cls) == 0:  # class 0 = id_card
                x1, y1, x2, y2 = map(int, box.xyxy[^2_0])
                return image[y1:y2, x1:x2]
        return image  # fallback: return full image

    def detect_fields(self, card_image: np.ndarray) -> dict:
        """Detect and crop individual fields"""
        results = self.model(card_image)[^2_0]
        fields = {}
        for box in results.boxes:
            cls = int(box.cls)
            label = self.field_labels.get(cls)
            if label:
                x1, y1, x2, y2 = map(int, box.xyxy[^2_0])
                fields[label] = card_image[y1:y2, x1:x2]
        return fields
```


### 3. `models/ocr_engine.py` — Qari-OCR + EasyOCR

```python
import easyocr
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import numpy as np

class OCREngine:
    def __init__(self):
        # EasyOCR للأرقام والإنجليزي
        self.easy_reader = easyocr.Reader(['ar', 'en'], gpu=torch.cuda.is_available())

        # Qari-OCR: أعلى دقة للعربي (NAMAA-Space - Jan 2026)
        self.qari_processor = AutoProcessor.from_pretrained("NAMAA-Space/Qari-OCR")
        self.qari_model = AutoModelForVision2Seq.from_pretrained(
            "NAMAA-Space/Qari-OCR",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.qari_model.to(self.device)

    def ocr_arabic_field(self, image_np: np.ndarray) -> str:
        """Use Qari-OCR for Arabic fields (name, address)"""
        image_pil = Image.fromarray(image_np).convert("RGB")
        inputs = self.qari_processor(images=image_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.qari_model.generate(**inputs, max_new_tokens=100)
        return self.qari_processor.decode(output[^2_0], skip_special_tokens=True).strip()

    def ocr_number_field(self, image_np: np.ndarray) -> str:
        """Use EasyOCR for ID number (digits only)"""
        results = self.easy_reader.readtext(image_np, detail=0, allowlist='0123456789')
        return ''.join(results).strip()

    def ocr_mixed_field(self, image_np: np.ndarray) -> str:
        """Use EasyOCR for mixed Arabic/English"""
        results = self.easy_reader.readtext(image_np, detail=0)
        return ' '.join(results).strip()
```


### 4. `models/id_parser.py` — تحليل الرقم القومي

```python
GOVERNORATES = {
    1: "القاهرة", 2: "الإسكندرية", 3: "بورسعيد", 4: "السويس",
    11: "دمياط", 12: "الدقهلية", 13: "الشرقية", 14: "القليوبية",
    15: "كفر الشيخ", 16: "الغربية", 17: "المنوفية", 18: "البحيرة",
    19: "الإسماعيلية", 21: "الجيزة", 22: "بني سويف", 23: "الفيوم",
    24: "المنيا", 25: "أسيوط", 26: "سوهاج", 27: "قنا", 28: "أسوان",
    29: "الأقصر", 31: "البحر الأحمر", 32: "الوادي الجديد",
    33: "مطروح", 34: "شمال سيناء", 35: "جنوب سيناء", 88: "خارج الجمهورية"
}

def parse_national_id(national_id: str) -> dict:
    national_id = ''.join(filter(str.isdigit, national_id))
    if len(national_id) != 14:
        return {"error": f"رقم غير صحيح، الطول: {len(national_id)}"}

    century_code = int(national_id[^2_0])
    year = int(national_id[1:3])
    month = int(national_id[3:5])
    day = int(national_id[5:7])
    gov_code = int(national_id[7:9])
    gender_digit = int(national_id[^2_12])

    century = 1900 if century_code == 2 else 2000

    return {
        "birth_date": f"{day:02d}/{month:02d}/{century + year}",
        "governorate": GOVERNORATES.get(gov_code, "غير معروف"),
        "gender": "ذكر" if gender_digit % 2 != 0 else "أنثى",
        "sequence": national_id[9:13],
        "checksum": national_id[^2_13]
    }
```


### 5. `utils/image_processing.py` — Preprocessing

```python
import cv2
import numpy as np

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Enhance image quality for better OCR"""
    # Resize to standard height
    h, w = image.shape[:2]
    if h < 100:
        scale = 100 / h
        image = cv2.resize(image, (int(w * scale), 100))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def deskew(image: np.ndarray) -> np.ndarray:
    """Correct image tilt"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
```


### 6. `app.py` — FastAPI Endpoint

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2, numpy as np
from models.detector import EgyptianIDDetector
from models.ocr_engine import OCREngine
from models.id_parser import parse_national_id
from utils.image_processing import preprocess_for_ocr, deskew

app = FastAPI(title="Egyptian ID OCR API", version="2.0")

detector = EgyptianIDDetector()
ocr = OCREngine()

@app.post("/extract")
async def extract_id_data(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="يجب رفع صورة فقط")

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    # Step 1: Deskew + crop card
    img = deskew(img)
    card = detector.detect_card(img)

    # Step 2: Detect fields
    fields = detector.detect_fields(card)

    # Step 3: OCR per field type
    result = {}
    arabic_fields = ["name", "address", "nationality"]
    number_fields = ["id_number"]

    for field_name, field_img in fields.items():
        processed = preprocess_for_ocr(field_img)
        if field_name in arabic_fields:
            result[field_name] = ocr.ocr_arabic_field(processed)
        elif field_name in number_fields:
            result[field_name] = ocr.ocr_number_field(processed)
        else:
            result[field_name] = ocr.ocr_mixed_field(processed)

    # Step 4: Parse national ID
    if "id_number" in result:
        result["parsed_info"] = parse_national_id(result["id_number"])

    return JSONResponse(content=result)

@app.get("/health")
def health():
    return {"status": "running", "models": ["YOLOv8", "Qari-OCR", "EasyOCR"]}
```


***

## تشغيل المشروع

```bash
# تشغيل الـ API
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Test بـ curl
curl -X POST "http://localhost:8000/extract" \
     -F "file=@egyptian_id.jpg"
```

**Response example:**

```json
{
  "name": "محمد كمال عبدالله",
  "address": "القاهرة، مدينة نصر",
  "id_number": "29901011234567",
  "nationality": "مصري",
  "parsed_info": {
    "birth_date": "01/01/1999",
    "governorate": "القاهرة",
    "gender": "ذكر"
  }
}
```


***

## ليه الـ Stack ده تحديداً؟

- **NASO7Y YOLO weights** — الموديل الوحيد المتاح مدرَّب specifically على Egyptian ID detection[^2_1]
- **Qari-OCR** — أحدث وأدق موديل Arabic OCR متاح على HuggingFace (يناير 2026)، يتفوق على EasyOCR في النص العربي بشكل واضح[^2_3]
- **EasyOCR للأرقام** — أسرع وأدق للأرقام اللاتينية والإنجليزية مقارنةً بالـ VLMs[^2_1]
- **FastAPI بدل Streamlit** — للـ production deployment ودمجه مع باقي مشاريعك بسهولة
<span style="display:none">[^2_10][^2_11][^2_14][^2_15][^2_5][^2_6][^2_7][^2_8][^2_9]</span>

<div align="center">⁂</div>

[^2_1]: https://github.com/NASO7Y/OCR_Egyptian_ID

[^2_2]: https://alpha-ollama.hf-mirror.com/loay/ArabicOCR-Qwen2.5-VL-7B-Vision

[^2_3]: https://huggingface.co/collections/NAMAA-Space/qari-ocr-a-high-accuracy-model-for-arabic-optical-character

[^2_4]: https://huggingface.co/papers/2509.18174

[^2_5]: https://m.sinosecu.com.cn/productDetail/0/47?lang=en_US

[^2_6]: https://www.youtube.com/shorts/nDUnVC87FlA

[^2_7]: https://kby-ai.com/4-real-life-id-document-ocr-challenges-in-processing/

[^2_8]: https://elibrary.stipram.ac.id/Record/doaj-5adf6ff8525043bb903b49e999400564

[^2_9]: https://github.com/Mostafa-Emad77/Egyptian-ID-Extraction

[^2_10]: https://github.com/huggingface/transformers/issues/19329

[^2_11]: https://www.codesota.com/ocr

[^2_12]: https://huggingface.co/MohamedRashad/arabic-large-nougat

[^2_13]: https://ejle.journals.ekb.eg/article_59924_98b54f20a36e88da1c58ca6fb99a1300.pdf

[^2_14]: https://signme.it/facial-recognition-from-egyptian-national-id/?lang=en

[^2_15]: https://huggingface.co/collections/MedoHamdani/arabic-ocr


---

# افضل architecture و افضل الطرق لزياده السرعه و الدقه و الفاعليه

الـ architecture المثالي لمشروع Egyptian ID OCR بيعتمد على **4 محاور**: Pipeline Design، Speed Optimization، Accuracy Optimization، و Reliability.

***

## الـ Architecture الأمثل

أفضل pattern ثبتت فاعليته في 2025 هو **Classify → Route → Specialize** — بدل موديل واحد بيعمل كل حاجة، بتصنف الـ field الأول وتوجهه للموديل المخصص له:[^3_1]

```
Input Image
     │
     ▼
┌─────────────────────────────────────────┐
│         PREPROCESSING LAYER             │
│  deskew → denoise → enhance → 300 DPI  │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│      YOLOv8-TensorRT (FP16)             │  ← ~4ms inference
│   Card Detection → Field Detection      │
└─────────────────────────────────────────┘
     │
     ├──────────────────────────────────────┐
     ▼                                      ▼
┌──────────────┐                  ┌──────────────────┐
│ Arabic OCR   │                  │  Numbers OCR     │
│  Qari-OCR    │                  │  EasyOCR (digits)│
│ (name/addr)  │                  │  (id_number)     │
└──────────────┘                  └──────────────────┘
     │                                      │
     └──────────────┬───────────────────────┘
                    ▼
        ┌─────────────────────┐
        │  Post-Processing    │
        │  + ID Parser        │
        │  + Confidence Score │
        └─────────────────────┘
                    │
                    ▼
             JSON Response
```


***

## 1. Speed Optimization

### TensorRT للـ YOLO (أهم خطوة)

TensorRT بيحوّل YOLO من ~15ms لـ ~4ms — **3.7x أسرع** بدون أي فقدان في الدقة:[^3_2][^3_3]

```python
from ultralytics import YOLO

model = YOLO("yolo_egyptian_id.pt")

# Export مرة واحدة بس
model.export(
    format="engine",   # TensorRT
    half=True,         # FP16 - نص الذاكرة، ضعف السرعة
    device=0,          # GPU
    workspace=4        # GB for optimization
)

# بعدين استخدم الـ engine دايماً
model_trt = YOLO("yolo_egyptian_id.engine")
results = model_trt(image, verbose=False)
```

| Precision | YOLO Time | FPS |
| :-- | :-- | :-- |
| FP32 (default) | 15.56ms | 52 |
| FP16 (PyTorch) | 10.35ms | 71 |
| **TensorRT FP16** | **4.11ms** | **127** |

[^3_2]

### Async Pipeline (FastAPI + asyncio)

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/extract")
async def extract_id(file: UploadFile = File(...)):
    contents = await file.read()

    # Run CPU-heavy OCR in thread pool (لا تبلوك الـ event loop)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, process_id_sync, contents)
    return result
```


### Image Caching بـ Redis

```python
import hashlib, redis, json

cache = redis.Redis(host="localhost", port=6379)

def get_cached_or_process(image_bytes: bytes) -> dict:
    img_hash = hashlib.md5(image_bytes).hexdigest()
    cached = cache.get(img_hash)
    if cached:
        return json.loads(cached)  # cache hit → 0ms

    result = full_pipeline(image_bytes)
    cache.setex(img_hash, 3600, json.dumps(result))  # TTL: 1 hour
    return result
```


***

## 2. Accuracy Optimization

### Smart Preprocessing (أكبر تأثير على الدقة)

الـ benchmark بيقول إن الـ preprocessing الصح بيرفع الدقة 15%:[^3_1]

```python
import cv2
import numpy as np

def advanced_preprocess(image: np.ndarray, target_dpi: int = 300) -> np.ndarray:
    # 1. Upscale إذا الصورة صغيرة (minimum 300 DPI equivalent)
    h, w = image.shape[:2]
    if h < 300:
        scale = 300 / h
        image = cv2.resize(image, (int(w * scale), 300),
                           interpolation=cv2.INTER_CUBIC)

    # 2. Denoise
    image = cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10)

    # 3. CLAHE - يحسن contrast في الإضاءة الضعيفة
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. Unsharp masking - يحسن حدة الحروف
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

    return image

def adaptive_threshold_for_ocr(gray: np.ndarray) -> np.ndarray:
    # Otsu threshold - automatically finds best threshold
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
```


### Hybrid OCR Voting (يقلل error rate 20%)

بدل موديل واحد، استخدم نتيجة أكثر من موديل وخد الأكثر confidence:[^3_1]

```python
from difflib import SequenceMatcher

def hybrid_ocr_vote(field_image: np.ndarray, field_type: str) -> tuple[str, float]:
    results = {}

    # Engine 1: EasyOCR
    easy_result = easy_reader.readtext(field_image, detail=1)
    if easy_result:
        text = ' '.join([r[^3_1] for r in easy_result])
        conf = np.mean([r[^3_2] for r in easy_result])
        results["easyocr"] = (text, conf)

    # Engine 2: Qari-OCR (للعربي فقط)
    if field_type in ["name", "address"]:
        qari_text = qari_ocr(field_image)
        results["qari"] = (qari_text, 0.9)  # Qari يثق به أكثر للعربي

    # اختار الأعلى confidence
    if not results:
        return "", 0.0

    best_engine = max(results, key=lambda k: results[k][^3_1])
    best_text, best_conf = results[best_engine]

    # Cross-validate: إذا الـ engines متفقين → confidence أعلى
    if len(results) > 1:
        texts = [v[^3_0] for v in results.values()]
        similarity = SequenceMatcher(None, texts[^3_0], texts[^3_1]).ratio()
        if similarity > 0.8:
            best_conf = min(best_conf * 1.1, 1.0)  # boost confidence

    return best_text, best_conf
```


### Post-Processing للنص العربي

```python
import re

def clean_arabic_text(text: str, field_type: str) -> str:
    # إزالة الحروف الغريبة
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\s\w]', '', text)
    text = ' '.join(text.split())  # normalize spaces

    if field_type == "id_number":
        # الرقم القومي: أرقام فقط، 14 خانة
        text = re.sub(r'\D', '', text)
        # تصحيح أخطاء شائعة في OCR للأرقام
        ocr_digit_fixes = {'O': '0', 'l': '1', 'I': '1', 'S': '5', 'B': '8'}
        for wrong, correct in ocr_digit_fixes.items():
            text = text.replace(wrong, correct)

    elif field_type == "name":
        # إزالة الأرقام من حقل الاسم
        text = re.sub(r'\d', '', text).strip()

    return text

def validate_national_id(id_str: str) -> bool:
    """Luhn-like validation للرقم القومي"""
    clean = re.sub(r'\D', '', id_str)
    if len(clean) != 14:
        return False
    century = int(clean[^3_0])
    if century not in [2, 3]:
        return False
    month = int(clean[3:5])
    day = int(clean[5:7])
    return 1 <= month <= 12 and 1 <= day <= 31
```


***

## 3. Reliability — Confidence Scoring

اعطي كل response درجة ثقة تخلي المستخدم يعرف إذا كانت النتيجة موثوقة:[^3_4]

```python
from enum import Enum

class ConfidenceLevel(str, Enum):
    HIGH = "high"       # > 0.85 → استخدم مباشرة
    MEDIUM = "medium"   # 0.6-0.85 → راجع يدوياً
    LOW = "low"         # < 0.6 → أعد الصورة

def build_response(extracted: dict, scores: dict) -> dict:
    overall_score = np.mean(list(scores.values()))

    return {
        "data": extracted,
        "confidence": {
            "overall": round(overall_score, 3),
            "level": (
                ConfidenceLevel.HIGH if overall_score > 0.85
                else ConfidenceLevel.MEDIUM if overall_score > 0.6
                else ConfidenceLevel.LOW
            ),
            "per_field": {k: round(v, 3) for k, v in scores.items()}
        },
        "validation": {
            "id_valid": validate_national_id(extracted.get("id_number", "")),
            "fields_found": list(extracted.keys())
        }
    }
```


***

## 4. Production Architecture الكاملة

```
                    ┌─────────────┐
                    │   Nginx     │  ← rate limiting
                    │ (Reverse    │
                    │  Proxy)     │
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │     FastAPI App         │
              │   (Async Workers)       │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │  Redis   │     │  YOLO    │     │  OCR     │
   │  Cache   │     │TensorRT  │     │ Engines  │
   └──────────┘     └──────────┘     └──────────┘
         │
         ▼
   ┌──────────┐
   │PostgreSQL│  ← audit log + results history
   └──────────┘
```


***

## ملخص الـ Performance Gains المتوقعة

| التحسين | نوعه | الكسب المتوقع |
| :-- | :-- | :-- |
| TensorRT FP16 | سرعة | **3.7x أسرع** inference [^3_2] |
| Async Workers | سرعة | يتعامل مع **4x طلبات** في نفس الوقت |
| Redis Cache | سرعة | **0ms** للصور المكررة |
| Advanced Preprocessing | دقة | **+15%** accuracy [^3_1] |
| Hybrid OCR Voting | دقة | **-20% error rate** [^3_1] |
| Post-Processing + Validation | موثوقية | يلغي نتائج غلط قبل ما توصل للمستخدم |

<span style="display:none">[^3_10][^3_11][^3_12][^3_13][^3_14][^3_15][^3_5][^3_6][^3_7][^3_8][^3_9]</span>

<div align="center">⁂</div>

[^3_1]: https://sparkco.ai/blog/2025-ocr-accuracy-benchmark-results-a-deep-dive-analysis

[^3_2]: https://blogjtr.com/posts/optimizing-perception/

[^3_3]: https://www.ultralytics.com/blog/optimizing-ultralytics-yolo-models-with-the-tensorrt-integration

[^3_4]: https://visionparser.com/blog/designing-ocr-pipelines-95-accuracy-visionparser

[^3_5]: https://bix-tech.com/ocr-in-2025-how-intelligent-ocr-turns-documents-into-data-use-cases-tools-and-best-practices/

[^3_6]: https://unstract.com/blog/best-opensource-ocr-tools-in-2025/

[^3_7]: https://aclanthology.org/2025.acl-long.749.pdf

[^3_8]: https://modal.com/blog/8-top-open-source-ocr-models-compared

[^3_9]: https://github.com/ultralytics/ultralytics/issues/13902

[^3_10]: https://intuitionlabs.ai/articles/non-llm-ocr-technologies

[^3_11]: https://www.youtube.com/watch?v=-qPGdGoh9Wg

[^3_12]: https://www.firstsource.com/insights/blogs/reading-understanding-how-ai-visual-processing-outperforms-traditional-ocr-complex

[^3_13]: https://www.reddit.com/r/LocalLLaMA/comments/1oppykf/is_ocr_accuracy_actually_a_blocker_for_anyones/

[^3_14]: https://github.com/orgs/ultralytics/discussions/8071

[^3_15]: https://healthedge.com/resources/blog/building-a-scalable-ocr-pipeline-technical-architecture-behind-healthedge-s-document-processing-platform


---

# لو عاوز اشغله علي cpu \& offline

للـ CPU Offline deployment، القاعدة الأساسية هي: **استبدل كل PyTorch بـ ONNX Runtime + quantization** — ده بيدي أسرع inference على CPU بدون أي dependency على GPU.[^4_1][^4_2]

***

## الاستراتيجية الكاملة للـ CPU

### المبدأ: ONNX + INT8 Quantization

```
PyTorch (.pt)  →  ONNX (.onnx)  →  INT8 Quantization
   ~80ms CPU         ~27ms CPU         ~12ms CPU
```

ONNX Runtime على CPU أسرع من PyTorch بشكل ثابت، وإضافة INT8 quantization بتوصل لـ ~3-4x أسرع من الأصل.[^4_3][^4_1]

***

## 1. Export YOLO لـ ONNX مع OpenVINO (الأسرع على Intel CPU)

```python
from ultralytics import YOLO

model = YOLO("yolo_egyptian_id.pt")

# Option A: ONNX (شغال على كل CPU)
model.export(
    format="onnx",
    imgsz=640,
    optimize=True,    # تحسينات إضافية للـ CPU
    half=False,       # CPU مش بيدعم FP16
    int8=True,        # INT8 quantization → أسرع بكتير
    opset=12
)

# Option B: OpenVINO (الأسرع على Intel CPU تحديداً)
model.export(format="openvino", imgsz=640, int8=True)
```

> **نصيحة:** لو الجهاز Intel، استخدم OpenVINO — بيتفوق على ONNX Runtime بـ ~30% على Intel hardware.[^4_4]

***

## 2. Inference Engine — ONNX Runtime بدل PyTorch

```python
import onnxruntime as ort
import numpy as np
import cv2

class FastCPUDetector:
    def __init__(self, onnx_path: str):
        # إعداد ONNX Runtime للـ CPU
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4          # استخدم 4 CPU cores
        opts.inter_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]  # CPU فقط
        )
        self.input_name = self.session.get_inputs()[^4_0].name

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img[np.newaxis, :]  # add batch dim

    def detect(self, image: np.ndarray) -> list:
        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return self.postprocess(outputs[^4_0])

    def postprocess(self, output: np.ndarray, conf_thresh=0.5) -> list:
        detections = []
        for det in output[^4_0]:
            conf = float(det[^4_4])
            if conf > conf_thresh:
                x1, y1, x2, y2 = map(int, det[:4])
                cls = int(det[^4_5])
                detections.append({"bbox": [x1, y1, x2, y2], "class": cls, "conf": conf})
        return detections
```


***

## 3. OCR على CPU — استبدل Qari-OCR بـ EasyOCR Quantized

Qari-OCR هو VLM ثقيل جداً على CPU (ممكن يأخد دقائق) — على CPU استخدم EasyOCR مع INT8 quantization:[^4_3]

```python
import easyocr
import torch

class CPUOptimizedOCR:
    def __init__(self):
        # EasyOCR مع quantization تلقائي على CPU
        self.reader = easyocr.Reader(
            ['ar', 'en'],
            gpu=False,
            quantize=True,          # INT8 quantization → 2-3x أسرع [web:47]
            detector=True,
            recognizer=True,
            verbose=False,
            model_storage_directory="./models_cache"  # offline: تخزين محلي
        )

    def ocr_field(self, image: np.ndarray) -> tuple[str, float]:
        # DBNet18 أسرع من CRAFT على CPU
        results = self.reader.readtext(
            image,
            detail=1,
            decoder='greedy',        # أسرع من beamsearch
            beamWidth=1,
            batch_size=1,
            workers=0,               # single-threaded أسرع على CPU صغير
            paragraph=False
        )
        if not results:
            return "", 0.0
        text = ' '.join([r[^4_1] for r in results])
        conf = float(np.mean([r[^4_2] for r in results]))
        return text, conf
```


***

## 4. Model Caching (حاسم للـ Offline)

أهم مشكلة في الـ offline هي إن الموديلات تكون موجودة محلياً من أول تشغيل:[^4_3]

```python
import os
from pathlib import Path

MODEL_DIR = Path("./models_cache")
MODEL_DIR.mkdir(exist_ok=True)

# تحميل الموديلات مرة واحدة عند startup
# وتخزينها في الذاكرة طول عمر الـ app
class ModelManager:
    _instance = None
    _detector = None
    _ocr = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._detector = FastCPUDetector("weights/yolo_egyptian_id_int8.onnx")
            cls._ocr = CPUOptimizedOCR()
            print("✅ Models loaded once, cached in memory")
        return cls._instance

# في FastAPI: load عند بدء التشغيل
@app.on_event("startup")
async def startup_event():
    ModelManager.get_instance()  # load مرة واحدة
```


***

## 5. الـ Pipeline الكامل CPU-Optimized

```python
import time

def process_id_cpu(image_bytes: bytes) -> dict:
    start = time.time()

    # decode image
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    manager = ModelManager.get_instance()

    # Step 1: دبلوك الصورة + crop
    img = deskew(img)
    img = advanced_preprocess(img)

    # Step 2: ONNX detection (~12-25ms على CPU)
    detections = manager._detector.detect(img)
    card_crop = crop_card(img, detections)
    fields = crop_fields(card_crop, detections)

    # Step 3: OCR لكل field
    result = {}
    scores = {}
    for field_name, field_img in fields.items():
        processed = preprocess_for_ocr(field_img)
        text, conf = manager._ocr.ocr_field(processed)
        result[field_name] = clean_arabic_text(text, field_name)
        scores[field_name] = conf

    # Step 4: parse ID
    if "id_number" in result:
        result["parsed_info"] = parse_national_id(result["id_number"])

    result["processing_time_ms"] = round((time.time() - start) * 1000, 1)
    result["confidence"] = scores

    return result
```


***

## 6. Requirements للـ Offline الكامل

```bash
# requirements.txt — كل حاجة تشتغل offline
fastapi==0.115.0
uvicorn==0.30.0
onnxruntime==1.19.0        # CPU inference
easyocr==1.7.1             # Arabic OCR
opencv-python-headless==4.10.0  # بدون GUI
numpy==1.26.4
Pillow==10.4.0
python-multipart==0.0.9

# اختياري لـ Intel CPU:
openvino==2024.4.0
```

```bash
# تحميل الموديلات مرة واحدة (وهو online) وتخزينها
python -c "
import easyocr
reader = easyocr.Reader(['ar','en'], model_storage_directory='./models_cache')
print('Models downloaded and cached ✅')
"

# بعدين اشتغل offline بالكامل
uvicorn app:app --host 0.0.0.0 --port 8000
```


***

## توقعات الـ Performance على CPU

| الخطوة | وقت على CPU (بدون تحسين) | وقت بعد التحسين |
| :-- | :-- | :-- |
| YOLO Detection | ~80ms | **~12-25ms** (ONNX INT8) [^4_1] |
| EasyOCR (per field) | ~30s | **~3-8s** (quantized) [^4_3] |
| Preprocessing | ~50ms | **~20ms** |
| **Total per image** | **~3 دقائق** | **~30-60 ثانية** |

> ⚠️ **الحقيقة:** OCR على CPU بطيء بطبيعته مقارنةً بالـ GPU. لو محتاج أقل من 5 ثوان، الحل الوحيد هو إما **GPU** أو استبدال EasyOCR بـ **Tesseract** (أسرع على CPU لكن أقل دقة في العربي).[^4_5][^4_6]
<span style="display:none">[^4_10][^4_11][^4_12][^4_13][^4_14][^4_15][^4_7][^4_8][^4_9]</span>

<div align="center">⁂</div>

[^4_1]: https://arxiv.org/html/2510.09653v1

[^4_2]: https://www.reddit.com/r/ryzen/comments/1lzr7yq/yolov8_multimachine_benchmark_rtx_3090_vs_ryzen/

[^4_3]: https://deepwiki.com/JaidedAI/EasyOCR/8.3-performance-optimization

[^4_4]: https://github.com/ultralytics/ultralytics/issues/3052

[^4_5]: https://github.com/JaidedAI/EasyOCR/issues/108

[^4_6]: https://github.com/JaidedAI/EasyOCR/issues/358

[^4_7]: https://stackoverflow.com/questions/73431716/i-am-trying-to-run-inference-on-a-single-gpu-with-easyocr-i-have-tried-running

[^4_8]: https://huggingface.co/asmud/EasyOCR-onnx

[^4_9]: https://blog.roboflow.com/how-to-use-easyocr/

[^4_10]: https://www.youtube.com/watch?v=ExT9QgQdVBw

[^4_11]: https://discuss.ray.io/t/increase-efficiency-using-pytorch-gpu-for-inference/6858

[^4_12]: https://github.com/ultralytics/ultralytics/issues/19499

[^4_13]: https://www.reddit.com/r/MachineLearning/comments/180kgh5/d_ways_to_speed_up_easyocrgpu_for_digits_only/

[^4_14]: https://pypi.org/project/easyocr/

[^4_15]: https://github.com/ultralytics/ultralytics/issues/6050


---

# full implementation plan in details

هنبني خطة تنفيذ كاملة ومرحلية، من الصفر لحد production — بأفضل stack لـ CPU offline.[^5_1][^5_2]

***

## Full Implementation Plan — Egyptian ID OCR (CPU / Offline)


***

## Phase 0 — Project Setup

```bash
# هيكل المشروع الكامل
egyptian_id_ocr/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI entry point
│   ├── api/
│   │   └── routes.py            # endpoints
│   ├── core/
│   │   ├── config.py            # settings
│   │   └── logger.py            # logging setup
│   ├── models/
│   │   ├── detector.py          # YOLO ONNX
│   │   ├── ocr_engine.py        # PaddleOCR + EasyOCR
│   │   └── id_parser.py         # national ID decoder
│   ├── services/
│   │   └── pipeline.py          # orchestrates the full flow
│   └── utils/
│       ├── image_utils.py       # preprocessing
│       └── text_utils.py        # postprocessing + cleaning
├── weights/
│   ├── yolo_card_detect.onnx    # YOLO model
│   └── yolo_fields_detect.onnx
├── models_cache/                # EasyOCR / PaddleOCR weights
├── tests/
│   ├── test_pipeline.py
│   └── sample_ids/
├── scripts/
│   ├── download_models.py       # تحميل الموديلات مرة واحدة
│   └── export_yolo_onnx.py      # convert YOLO → ONNX
├── requirements.txt
├── Dockerfile
└── README.md
```

```bash
# إنشاء البيئة
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install fastapi uvicorn[standard] \
    onnxruntime opencv-python-headless \
    easyocr paddlepaddle paddleocr \
    pillow numpy python-multipart \
    pydantic-settings loguru pytest httpx
```


***

## Phase 1 — Model Preparation

### Step 1.1 — تحميل YOLO weights من NASO7Y

```bash
# clone المشروع الأصلي
git clone https://github.com/NASO7Y/OCR_Egyptian_ID.git
# خد الـ weights منه واحطها في /weights
```


### Step 1.2 — Export YOLO → ONNX INT8

```python
# scripts/export_yolo_onnx.py
from ultralytics import YOLO

for model_name, output_name in [
    ("weights/yolo_card_detect.pt", "weights/yolo_card_detect.onnx"),
    ("weights/yolo_fields_detect.pt", "weights/yolo_fields_detect.onnx"),
]:
    model = YOLO(model_name)
    model.export(
        format="onnx",
        imgsz=640,
        optimize=True,
        opset=12,
        simplify=True    # simplify graph → أسرع inference
    )
    print(f"✅ Exported: {output_name}")
```


### Step 1.3 — تحميل PaddleOCR + EasyOCR مرة واحدة

```python
# scripts/download_models.py
# شغّله مرة واحدة وهو online

# PaddleOCR — أفضل OCR لـ Arabic على CPU في 2025
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="arabic",
    use_gpu=False,
    det_model_dir="./models_cache/paddle_det",
    rec_model_dir="./models_cache/paddle_rec",
    cls_model_dir="./models_cache/paddle_cls",
)
print("✅ PaddleOCR models downloaded")

# EasyOCR — للأرقام والإنجليزي
import easyocr
reader = easyocr.Reader(['ar', 'en'], gpu=False,
                         model_storage_directory="./models_cache/easyocr")
print("✅ EasyOCR models downloaded")
```

> **ليه PaddleOCR للعربي؟** PP-OCRv5 يدعم 109 لغة بـ 2M parameter فقط، وبيشتغل على CPU بـ ~21ms per field — أسرع وأدق من EasyOCR للعربي على CPU.[^5_3][^5_1]

***

## Phase 2 — Core Modules

### `app/core/config.py`

```python
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Paths
    YOLO_CARD_MODEL: str = "weights/yolo_card_detect.onnx"
    YOLO_FIELDS_MODEL: str = "weights/yolo_fields_detect.onnx"
    MODELS_CACHE_DIR: str = "./models_cache"

    # Inference
    YOLO_CONF_THRESHOLD: float = 0.5
    YOLO_IOU_THRESHOLD: float = 0.45
    YOLO_INPUT_SIZE: int = 640
    OCR_CPU_THREADS: int = 4

    # Field class IDs (حسب training الـ YOLO)
    CLASS_ID_CARD: int = 0
    CLASS_NAMES: dict = {
        1: "name_ar", 2: "name_en", 3: "address",
        4: "id_number", 5: "birth_date",
        6: "gender", 7: "nationality"
    }

    class Config:
        env_file = ".env"

settings = Settings()
```


### `app/core/logger.py`

```python
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")
logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="DEBUG")
```


***

### `app/models/detector.py` — ONNX YOLO

```python
import onnxruntime as ort
import numpy as np
import cv2
from app.core.config import settings
from app.core.logger import logger

class ONNXDetector:
    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = settings.OCR_CPU_THREADS
        opts.inter_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[^5_0].name
        self.input_size = settings.YOLO_INPUT_SIZE
        logger.info(f"✅ Loaded ONNX model: {model_path}")

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = image.shape[:2]
        scale_x, scale_y = w / self.input_size, h / self.input_size

        resized = cv2.resize(image, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return tensor[np.newaxis, :], scale_x, scale_y

    def postprocess(self, output: np.ndarray, scale_x: float, scale_y: float) -> list:
        detections = []
        preds = output[^5_0]
        for pred in preds:
            conf = float(pred[^5_4])
            if conf < settings.YOLO_CONF_THRESHOLD:
                continue
            cx, cy, bw, bh = pred[:4]
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)
            cls = int(np.argmax(pred[5:]))
            detections.append({
                "bbox": [max(0,x1), max(0,y1), x2, y2],
                "class_id": cls,
                "confidence": conf
            })
        return detections

    def run(self, image: np.ndarray) -> list:
        tensor, sx, sy = self.preprocess(image)
        output = self.session.run(None, {self.input_name: tensor})
        return self.postprocess(output[^5_0], sx, sy)


class EgyptianIDDetector:
    def __init__(self):
        self.card_detector = ONNXDetector(settings.YOLO_CARD_MODEL)
        self.field_detector = ONNXDetector(settings.YOLO_FIELDS_MODEL)

    def crop_card(self, image: np.ndarray) -> np.ndarray:
        detections = self.card_detector.run(image)
        card_dets = [d for d in detections if d["class_id"] == settings.CLASS_ID_CARD]
        if not card_dets:
            logger.warning("No card detected, using full image")
            return image
        best = max(card_dets, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = best["bbox"]
        return image[y1:y2, x1:x2]

    def crop_fields(self, card_image: np.ndarray) -> dict[str, np.ndarray]:
        detections = self.field_detector.run(card_image)
        fields = {}
        for det in detections:
            label = settings.CLASS_NAMES.get(det["class_id"])
            if label:
                x1, y1, x2, y2 = det["bbox"]
                crop = card_image[y1:y2, x1:x2]
                if crop.size > 0:
                    fields[label] = (crop, det["confidence"])
        return fields
```


***

### `app/models/ocr_engine.py` — PaddleOCR + EasyOCR

```python
import numpy as np
import easyocr
from paddleocr import PaddleOCR
from app.core.config import settings
from app.core.logger import logger

class OCREngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_models()
        return cls._instance

    def _init_models(self):
        logger.info("Loading OCR models...")

        # PaddleOCR — أفضل للعربي على CPU
        self.paddle = PaddleOCR(
            use_angle_cls=True,
            lang="arabic",
            use_gpu=False,
            cpu_threads=settings.OCR_CPU_THREADS,
            det_model_dir=f"{settings.MODELS_CACHE_DIR}/paddle_det",
            rec_model_dir=f"{settings.MODELS_CACHE_DIR}/paddle_rec",
            cls_model_dir=f"{settings.MODELS_CACHE_DIR}/paddle_cls",
            show_log=False,
            enable_mkldnn=True,   # Intel MKL-DNN → ~2x أسرع على Intel CPU
        )

        # EasyOCR — للأرقام والإنجليزي
        self.easy = easyocr.Reader(
            ['ar', 'en'],
            gpu=False,
            quantize=True,
            model_storage_directory=f"{settings.MODELS_CACHE_DIR}/easyocr",
            verbose=False
        )
        logger.info("✅ OCR models ready")

    def ocr_arabic(self, image: np.ndarray) -> tuple[str, float]:
        """PaddleOCR للحقول العربية"""
        try:
            result = self.paddle.ocr(image, cls=True)
            if not result or not result[^5_0]:
                return "", 0.0
            texts = [line[^5_1][^5_0] for line in result[^5_0]]
            confs = [line[^5_1][^5_1] for line in result[^5_0]]
            return ' '.join(texts), float(np.mean(confs))
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return "", 0.0

    def ocr_digits(self, image: np.ndarray) -> tuple[str, float]:
        """EasyOCR للأرقام فقط"""
        try:
            results = self.easy.readtext(
                image,
                detail=1,
                allowlist='0123456789',
                decoder='greedy',
                beamWidth=1
            )
            if not results:
                return "", 0.0
            text = ''.join([r[^5_1] for r in results])
            conf = float(np.mean([r[^5_2] for r in results]))
            return text, conf
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return "", 0.0

    def ocr_mixed(self, image: np.ndarray) -> tuple[str, float]:
        """PaddleOCR للحقول المختلطة"""
        return self.ocr_arabic(image)
```


***

### `app/models/id_parser.py`

```python
import re

GOVERNORATES = {
    1:"القاهرة", 2:"الإسكندرية", 3:"بورسعيد", 4:"السويس",
    11:"دمياط", 12:"الدقهلية", 13:"الشرقية", 14:"القليوبية",
    15:"كفر الشيخ", 16:"الغربية", 17:"المنوفية", 18:"البحيرة",
    19:"الإسماعيلية", 21:"الجيزة", 22:"بني سويف", 23:"الفيوم",
    24:"المنيا", 25:"أسيوط", 26:"سوهاج", 27:"قنا",
    28:"أسوان", 29:"الأقصر", 31:"البحر الأحمر",
    32:"الوادي الجديد", 33:"مطروح", 34:"شمال سيناء",
    35:"جنوب سيناء", 88:"خارج الجمهورية"
}

def parse_national_id(raw: str) -> dict:
    nid = re.sub(r'\D', '', raw)

    # تصحيح أخطاء OCR الشائعة
    for wrong, right in [('O','0'),('o','0'),('l','1'),('I','1'),('S','5')]:
        nid = nid.replace(wrong, right)

    if len(nid) != 14:
        return {"valid": False, "error": f"Expected 14 digits, got {len(nid)}"}

    century_code = int(nid[^5_0])
    if century_code not in [2, 3]:
        return {"valid": False, "error": "Invalid century code"}

    year = int(nid[1:3])
    month = int(nid[3:5])
    day = int(nid[5:7])
    gov_code = int(nid[7:9])
    gender_digit = int(nid[^5_12])
    century = 1900 if century_code == 2 else 2000

    if not (1 <= month <= 12 and 1 <= day <= 31):
        return {"valid": False, "error": "Invalid birth date"}

    return {
        "valid": True,
        "birth_date": f"{day:02d}/{month:02d}/{century + year}",
        "governorate": GOVERNORATES.get(gov_code, "غير معروف"),
        "gender": "ذكر" if gender_digit % 2 != 0 else "أنثى",
        "age": 2026 - (century + year),
        "sequence": nid[9:13],
        "raw": nid
    }
```


***

### `app/utils/image_utils.py`

```python
import cv2
import numpy as np

def deskew(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) < 10:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = 90 + angle
    if abs(angle) < 0.5: return image   # skip tiny rotations
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    # Upscale if too small
    h, w = image.shape[:2]
    if h < 60:
        scale = 60 / h
        image = cv2.resize(image, (int(w*scale), 60),
                           interpolation=cv2.INTER_CUBIC)

    # CLAHE contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    image = cv2.filter2D(image, -1, kernel)

    return image

def decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    return img
```


### `app/utils/text_utils.py`

```python
import re

ARABIC_RANGE = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'

def clean_field(text: str, field_type: str) -> str:
    text = text.strip()

    if field_type == "id_number":
        text = re.sub(r'\D', '', text)
        # Fix common OCR digit mistakes
        for w, r in [('O','0'),('o','0'),('l','1'),('I','1'),('S','5'),('B','8')]:
            text = text.replace(w, r)

    elif field_type in ["name_ar", "address", "nationality"]:
        # Keep only Arabic chars + spaces
        text = re.sub(f'[^{ARABIC_RANGE[1:-1]}\\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

    elif field_type == "name_en":
        text = re.sub(r'[^A-Za-z\s]', '', text).strip().upper()

    elif field_type in ["birth_date", "gender"]:
        pass  # keep as is

    return text
```


***

### `app/services/pipeline.py` — الـ Orchestrator

```python
import time
import numpy as np
from app.models.detector import EgyptianIDDetector
from app.models.ocr_engine import OCREngine
from app.models.id_parser import parse_national_id
from app.utils.image_utils import deskew, enhance_for_ocr, decode_image
from app.utils.text_utils import clean_field
from app.core.logger import logger

ARABIC_FIELDS = {"name_ar", "address", "nationality", "gender"}
DIGIT_FIELDS = {"id_number"}

class IDExtractionPipeline:
    _detector = None
    _ocr = None

    @classmethod
    def initialize(cls):
        logger.info("Initializing pipeline...")
        cls._detector = EgyptianIDDetector()
        cls._ocr = OCREngine()
        logger.info("✅ Pipeline ready")

    @classmethod
    def process(cls, image_bytes: bytes) -> dict:
        t0 = time.time()

        # 1. Decode + preprocess
        image = decode_image(image_bytes)
        image = deskew(image)

        # 2. Detect card
        card = cls._detector.crop_card(image)

        # 3. Detect fields
        fields = cls._detector.crop_fields(card)

        if not fields:
            return {"error": "No fields detected", "processing_ms": _ms(t0)}

        # 4. OCR per field
        extracted = {}
        confidence_scores = {}

        for field_name, (field_img, det_conf) in fields.items():
            enhanced = enhance_for_ocr(field_img)

            if field_name in DIGIT_FIELDS:
                text, ocr_conf = cls._ocr.ocr_digits(enhanced)
            elif field_name in ARABIC_FIELDS:
                text, ocr_conf = cls._ocr.ocr_arabic(enhanced)
            else:
                text, ocr_conf = cls._ocr.ocr_mixed(enhanced)

            cleaned = clean_field(text, field_name)
            extracted[field_name] = cleaned
            confidence_scores[field_name] = round(
                (det_conf + ocr_conf) / 2, 3
            )

        # 5. Parse national ID
        if "id_number" in extracted:
            extracted["parsed_id"] = parse_national_id(extracted["id_number"])

        # 6. Overall confidence
        avg_conf = round(float(np.mean(list(confidence_scores.values()))), 3)
        level = "high" if avg_conf > 0.85 else "medium" if avg_conf > 0.6 else "low"

        return {
            "extracted": extracted,
            "confidence": {
                "overall": avg_conf,
                "level": level,
                "per_field": confidence_scores
            },
            "processing_ms": _ms(t0)
        }

def _ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)
```


***

### `app/api/routes.py`

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.pipeline import IDExtractionPipeline

router = APIRouter(prefix="/api/v1", tags=["OCR"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_SIZE_MB = 10

@router.post("/extract")
async def extract_id(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported format: {file.content_type}")

    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_SIZE_MB}MB)")

    result = IDExtractionPipeline.process(contents)
    return JSONResponse(content=result)

@router.get("/health")
def health():
    return {"status": "ok", "mode": "CPU / Offline"}
```


### `app/main.py`

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.services.pipeline import IDExtractionPipeline
from app.core.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    IDExtractionPipeline.initialize()   # load once at startup
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Egyptian ID OCR API",
    version="1.0.0",
    description="Offline CPU-based Egyptian National ID extraction",
    lifespan=lifespan
)

app.include_router(router)
```


***

## Phase 3 — Tests

```python
# tests/test_pipeline.py
import pytest
from pathlib import Path
from app.services.pipeline import IDExtractionPipeline
from app.models.id_parser import parse_national_id

IDExtractionPipeline.initialize()

def test_id_parser_valid():
    result = parse_national_id("29901011234567")
    assert result["valid"] == True
    assert result["gender"] == "ذكر"
    assert result["governorate"] == "القاهرة"
    assert result["birth_date"] == "01/01/1999"

def test_id_parser_invalid():
    result = parse_national_id("12345")
    assert result["valid"] == False

def test_pipeline_with_sample():
    sample = Path("tests/sample_ids/test_id.jpg")
    if sample.exists():
        with open(sample, "rb") as f:
            result = IDExtractionPipeline.process(f.read())
        assert "extracted" in result
        assert result["confidence"]["overall"] > 0
        print(f"Result: {result}")

# تشغيل
# pytest tests/ -v
```


***

## Phase 4 — Docker (Offline Complete)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy models + weights (pre-downloaded)
COPY weights/ ./weights/
COPY models_cache/ ./models_cache/

# Copy app code
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

```bash
# Build + Run
docker build -t egyptian-id-ocr .
docker run -p 8000:8000 --cpus="4" --memory="4g" egyptian-id-ocr
```


***

## Phase 5 — خطوات التشغيل الكاملة

```bash
# 1. Clone + setup
git clone <your-repo>
cd egyptian_id_ocr
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. احضر الـ YOLO weights من NASO7Y
git clone https://github.com/NASO7Y/OCR_Egyptian_ID.git
cp OCR_Egyptian_ID/weights/*.pt weights/

# 3. Export YOLO → ONNX (مرة واحدة)
python scripts/export_yolo_onnx.py

# 4. تحميل OCR models (مرة واحدة، وهو online)
python scripts/download_models.py

# 5. تشغيل الـ API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 6. اختبار
curl -X POST http://localhost:8000/api/v1/extract \
     -F "file=@tests/sample_ids/test_id.jpg"
```


***

## مقارنة الـ Tech Choices النهائية

| القرار | الاختيار | السبب |
| :-- | :-- | :-- |
| Detection | YOLO ONNX INT8 | أسرع inference على CPU [^5_4] |
| Arabic OCR | PaddleOCR v4 | أسرع وأدق للعربي على CPU، MKL-DNN support [^5_3][^5_1] |
| Digits OCR | EasyOCR (quantized) | دقيق للأرقام، lightweight [^5_2] |
| API Framework | FastAPI + lifespan | load models مرة واحدة |
| Deployment | Docker | reproducible offline environment |
| Logging | Loguru | أبسط وأقوى من logging الافتراضي |

<span style="display:none">[^5_10][^5_11][^5_13][^5_14][^5_15][^5_16][^5_5][^5_6][^5_7][^5_8][^5_9]</span>

<div align="center">⁂</div>

[^5_1]: https://github.com/PaddlePaddle/PaddleOCR

[^5_2]: https://www.linkedin.com/posts/alaa-taha-el-maria_ai-machinelearning-ocr-activity-7420141772376375296-ZH_p

[^5_3]: https://paddlepaddle.github.io/PaddleX/3.3/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html

[^5_4]: https://arxiv.org/html/2510.09653v1

[^5_5]: https://github.com/Mostafa-Emad77/Egyptian-ID-Extraction

[^5_6]: https://ejle.journals.ekb.eg/article_59924_98b54f20a36e88da1c58ca6fb99a1300.pdf

[^5_7]: https://arya.ai/apex-apis/egypt-national-id-api

[^5_8]: https://www.studocu.com/row/document/new-mansoura-university/algorithms/ejle-volume-1-issue-2-pages-35-49/103937201

[^5_9]: https://signme.it/egypt-national-id-api/?lang=en

[^5_10]: https://repository.rit.edu/theses/12449/

[^5_11]: https://dev.to/czmilo/2025-complete-guide-paddleocr-vl-09b-baidus-ultra-lightweight-document-parsing-powerhouse-1e8l

[^5_12]: https://learn.microsoft.com/en-us/answers/questions/443478/best-practices-for-arabic-ocr?page=1

[^5_13]: https://huggingface.co/PaddlePaddle/PaddleOCR-VL

[^5_14]: https://ae.linkedin.com/posts/mohamed-el-tahawey-7338aa144_github-tahaweyidocr-egyptian-id-recognition-activity-7105197370451128321-phAw

[^5_15]: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html

[^5_16]: https://regulaforensics.com/blog/arabic-script-id-documents-processing/


---

# كود كامل للـ preprocessing واستخراج الـ ROIs للبطاقة المصرية

هنبني ملف `image_utils.py` كامل ومتكامل يغطي كل مراحل الـ preprocessing من أول ما الصورة تيجي لحد ما الـ ROIs تكون جاهزة للـ OCR.[^6_1][^6_2][^6_3]

***

## `app/utils/image_utils.py` — الكود الكامل

```python
"""
Egyptian ID Card - Full Preprocessing & ROI Extraction Pipeline
يغطي: تحسين الصورة، تصحيح الزاوية، Perspective Transform، واستخراج الـ ROIs
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from app.core.logger import logger


# ─────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────

@dataclass
class CardROIs:
    """كل الـ ROIs المستخرجة من البطاقة"""
    full_card: np.ndarray          # البطاقة كاملة بعد التصحيح
    name_ar: Optional[np.ndarray] = None        # الاسم بالعربي
    name_en: Optional[np.ndarray] = None        # الاسم بالإنجليزي
    address: Optional[np.ndarray] = None        # العنوان
    id_number: Optional[np.ndarray] = None      # الرقم القومي
    birth_date: Optional[np.ndarray] = None     # تاريخ الميلاد
    gender: Optional[np.ndarray] = None         # النوع
    nationality: Optional[np.ndarray] = None    # الجنسية
    face: Optional[np.ndarray] = None           # الصورة الشخصية
    quality_score: float = 0.0                  # جودة الصورة (0-1)


# ─────────────────────────────────────────────────────────
# Step 1: Image Quality Assessment
# ─────────────────────────────────────────────────────────

def assess_quality(image: np.ndarray) -> dict:
    """
    تقييم جودة الصورة قبل أي معالجة
    Returns: dict بـ scores و overall quality
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness — Laplacian variance (كلما أعلى كلما أوضح)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(laplacian_var / 500.0, 1.0)  # normalize to 0-1

    # 2. Brightness — mean pixel value
    brightness = float(np.mean(gray)) / 255.0
    brightness_score = 1.0 - abs(brightness - 0.5) * 2  # أحسن لو قريب من 0.5

    # 3. Contrast — standard deviation
    contrast = float(np.std(gray)) / 128.0
    contrast_score = min(contrast, 1.0)

    # 4. Glare detection — check for blown-out highlights
    overexposed = np.sum(gray > 240) / gray.size
    glare_score = 1.0 - min(overexposed * 10, 1.0)

    overall = np.mean([sharpness, brightness_score, contrast_score, glare_score])

    return {
        "overall": round(float(overall), 3),
        "sharpness": round(float(sharpness), 3),
        "brightness": round(float(brightness_score), 3),
        "contrast": round(float(contrast_score), 3),
        "glare": round(float(glare_score), 3),
        "acceptable": overall > 0.35  # minimum threshold
    }


# ─────────────────────────────────────────────────────────
# Step 2: Basic Preprocessing
# ─────────────────────────────────────────────────────────

def resize_to_standard(image: np.ndarray, target_width: int = 1200) -> np.ndarray:
    """Resize للعرض المعياري مع الحفاظ على الـ aspect ratio"""
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / w
    new_h = int(h * scale)
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (target_width, new_h), interpolation=interp)


def remove_noise(image: np.ndarray) -> np.ndarray:
    """إزالة الضوضاء مع الحفاظ على حواف الحروف"""
    # Bilateral filter: يحافظ على الحواف أحسن من Gaussian
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """تحسين التباين باستخدام CLAHE على قناة L في LAB"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpen_image(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Unsharp masking — يحسن حدة الحروف"""
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    return cv2.addWeighted(image, 1 + strength, gaussian, -strength, 0)


# ─────────────────────────────────────────────────────────
# Step 3: Card Detection & Perspective Correction
# ─────────────────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    """ترتيب 4 نقاط: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[^6_0] = pts[np.argmin(s)]   # top-left: أصغر مجموع
    rect[^6_2] = pts[np.argmax(s)]   # bottom-right: أكبر مجموع
    diff = np.diff(pts, axis=1)
    rect[^6_1] = pts[np.argmin(diff)]  # top-right: أصغر فرق
    rect[^6_3] = pts[np.argmax(diff)]  # bottom-left: أكبر فرق
    return rect


def perspective_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Perspective Transform — يصحح زاوية التصوير ويعطي صورة مستقيمة
    كأنك صورت البطاقة من فوق مباشرة
    """
    rect = order_points(corners)
    tl, tr, br, bl = rect

    # حساب العرض والارتفاع الجديد
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def find_card_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """
    إيجاد حواف البطاقة في الصورة باستخدام Contour Detection
    Returns: numpy array بـ 4 نقاط أو None لو مش لاقي
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocessing للـ edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Dilate عشان نوصل الـ edges المنفصلة
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)

    # إيجاد الـ contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    img_area = image.shape[^6_0] * image.shape[^6_1]

    for contour in contours:
        area = cv2.contourArea(contour)

        # البطاقة لازم تكون على الأقل 15% من الصورة
        if area < img_area * 0.15:
            continue

        # تقريب الـ contour لشكل مضلع
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # لو وجدنا مستطيل (4 نقاط)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    logger.warning("Could not find card corners via contour detection")
    return None


def auto_detect_and_warp(image: np.ndarray) -> np.ndarray:
    """
    تلقائياً: اكتشف البطاقة + صحح الـ perspective
    لو مش قادر يكتشف → يرجع الصورة كما هي
    """
    resized = resize_to_standard(image, 1200)
    corners = find_card_corners(resized)

    if corners is not None:
        warped = perspective_transform(resized, corners)
        logger.info(f"✅ Perspective corrected: {warped.shape}")
        return warped

    # Fallback: deskew بس
    logger.info("Using deskew fallback")
    return deskew(resized)


def deskew(image: np.ndarray) -> np.ndarray:
    """
    تصحيح الانحراف الطفيف (rotation) باستخدام Projection Profile
    أفضل للصور المسحوبة scan مع انحراف بسيط [web:23]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # Threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[^6_1]
    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 50:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle

    # تجاهل الزوايا الصغيرة جداً (أقل من 0.5 درجة)
    if abs(angle) < 0.5:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    logger.info(f"Deskewed by {angle:.2f}°")
    return rotated


# ─────────────────────────────────────────────────────────
# Step 4: ROI Extraction (Relative Coordinates)
# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
# البطاقة المصرية layout (relative coords بالنسبة للبطاقة كلها)
# الإحداثيات: (x_ratio, y_ratio, w_ratio, h_ratio)
# Front side — الوجه الأمامي
# ─────────────────────────────────────────────────────────

FRONT_ROIS = {
    # الصورة الشخصية (يسار البطاقة)
    "face":         (0.02, 0.08, 0.28, 0.75),

    # الاسم بالعربي (سطرين في الغالب)
    "name_ar":      (0.32, 0.08, 0.66, 0.22),

    # الاسم بالإنجليزي (تحت الاسم العربي)
    "name_en":      (0.32, 0.30, 0.66, 0.14),

    # الجنس والجنسية (في نفس السطر)
    "gender":       (0.32, 0.44, 0.30, 0.13),
    "nationality":  (0.62, 0.44, 0.36, 0.13),

    # تاريخ الميلاد
    "birth_date":   (0.32, 0.57, 0.50, 0.13),

    # الرقم القومي (أسفل البطاقة)
    "id_number":    (0.10, 0.82, 0.80, 0.14),
}

BACK_ROIS = {
    # العنوان (عادةً في أعلى الوجه الخلفي)
    "address":      (0.05, 0.08, 0.90, 0.35),
    "job":          (0.05, 0.44, 0.60, 0.15),
    "release_date": (0.05, 0.60, 0.45, 0.15),
    "expiry_date":  (0.55, 0.60, 0.40, 0.15),
}


def extract_roi(image: np.ndarray, roi: tuple) -> np.ndarray:
    """استخراج ROI واحد من الصورة"""
    h, w = image.shape[:2]
    x_r, y_r, w_r, h_r = roi
    x = int(x_r * w)
    y = int(y_r * h)
    rw = int(w_r * w)
    rh = int(h_r * h)

    # Padding صغير حول الـ ROI
    pad = 4
    x = max(0, x - pad)
    y = max(0, y - pad)
    rw = min(w - x, rw + pad * 2)
    rh = min(h - y, rh + pad * 2)

    crop = image[y:y+rh, x:x+rw]
    return crop if crop.size > 0 else image


def extract_all_rois(card_image: np.ndarray, side: str = "front") -> dict[str, np.ndarray]:
    """استخرج كل الـ ROIs من البطاقة"""
    rois_map = FRONT_ROIS if side == "front" else BACK_ROIS
    return {
        field: extract_roi(card_image, coords)
        for field, coords in rois_map.items()
    }


# ─────────────────────────────────────────────────────────
# Step 5: Per-Field Preprocessing for OCR
# ─────────────────────────────────────────────────────────

def preprocess_text_field(image: np.ndarray, field_type: str = "arabic") -> np.ndarray:
    """
    معالجة متخصصة لكل نوع حقل قبل الـ OCR
    """
    # تحويل لـ grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Upscale إذا صغير (minimum 64px height)
    h, w = gray.shape
    if h < 64:
        scale = 64 / h
        gray = cv2.resize(gray, (int(w * scale), 64),
                          interpolation=cv2.INTER_CUBIC)

    if field_type == "id_number":
        # للرقم القومي: threshold حاد + morphological cleanup
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # إزالة dots وضوضاء صغيرة
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    elif field_type in ["arabic", "name_ar", "address", "nationality", "gender"]:
        # للعربي: Adaptive threshold أفضل للخلفيات غير المنتظمة [web:74]
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8
        )
        # Dilation طفيف يوصل حروف العربي المنفصلة
        kernel = np.ones((1, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        return binary

    elif field_type in ["name_en", "birth_date"]:
        # للإنجليزي والتواريخ: Otsu threshold
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    elif field_type == "face":
        # الصورة الشخصية: ما نعملش binarization
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image

    else:
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary


def remove_background_lines(image: np.ndarray) -> np.ndarray:
    """
    إزالة الخطوط الأفقية والعمودية الموجودة في خلفية البطاقة
    يحسن دقة الـ OCR بشكل ملحوظ [web:23]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[^6_1]

    # إزالة الخطوط الأفقية
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)

    # إزالة الخطوط العمودية
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # دمج الخطوط وطرحهم من الصورة
    lines_mask = cv2.add(h_lines, v_lines)
    result = cv2.subtract(thresh, lines_mask)

    return cv2.bitwise_not(result)


# ─────────────────────────────────────────────────────────
# Step 6: Side Detection (Front vs Back)
# ─────────────────────────────────────────────────────────

def detect_card_side(image: np.ndarray) -> str:
    """
    تحديد هل البطاقة وجه أمامي أم خلفي
    بيعتمد على: وجود الصورة الشخصية (circle/oval region) للأمامي
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # تفحص المنطقة اليسارية (حيث الصورة الشخصية في الأمامي)
    left_region = gray[:, :int(w * 0.30)]

    # إيجاد circles (الصورة الشخصية غالباً دايرة أو شبه دايرة)
    circles = cv2.HoughCircles(
        left_region,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=100
    )

    if circles is not None:
        return "front"

    # طريقة بديلة: تفحص الكثافة اللونية في منطقة الصورة
    face_region = image[:, :int(w * 0.30)]
    face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    variance = np.var(face_gray)

    # الصورة الشخصية = variance عالي (وجه بيفرق)
    return "front" if variance > 500 else "back"


# ─────────────────────────────────────────────────────────
# Step 7: Main Pipeline Entry Point
# ─────────────────────────────────────────────────────────

def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes to numpy array"""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image — invalid format or corrupted file")
    return img


def full_preprocess_pipeline(
    image: np.ndarray,
    use_yolo: bool = False,
    yolo_fields: Optional[dict] = None
) -> CardROIs:
    """
    Pipeline كامل من الصورة الخام لـ CardROIs جاهزة للـ OCR

    Args:
        image: الصورة الخام
        use_yolo: لو True استخدم YOLO للـ field detection
        yolo_fields: الـ fields المكتشفة من YOLO (لو use_yolo=True)

    Returns:
        CardROIs object بكل الحقول المستخرجة والمعالجة
    """
    # ── 1. Quality Assessment ──────────────────────────
    quality = assess_quality(image)
    logger.info(f"Image quality: {quality['overall']:.2f} | {quality}")

    if not quality["acceptable"]:
        logger.warning(f"Low quality image (score: {quality['overall']:.2f}), proceeding anyway")

    # ── 2. Resize + Basic Enhancement ──────────────────
    image = resize_to_standard(image, target_width=1200)
    image = remove_noise(image)
    image = enhance_contrast(image)

    # ── 3. Perspective Correction ──────────────────────
    card = auto_detect_and_warp(image)

    # ── 4. Final Sharpening ───────────────────────────
    card = sharpen_image(card, strength=1.2)

    # ── 5. Detect Side ────────────────────────────────
    side = detect_card_side(card)
    logger.info(f"Detected card side: {side}")

    # ── 6. Extract ROIs ──────────────────────────────
    if use_yolo and yolo_fields:
        # استخدم YOLO للـ precision العالية
        raw_rois = yolo_fields
    else:
        # استخدم الـ relative coordinates كـ fallback
        raw_rois = extract_all_rois(card, side=side)

    # ── 7. Remove Background Lines ───────────────────
    card_clean = remove_background_lines(card)

    # ── 8. Per-Field Preprocessing ───────────────────
    processed_rois = {}
    for field_name, roi_img in raw_rois.items():
        if roi_img is None or roi_img.size == 0:
            continue
        processed = preprocess_text_field(roi_img, field_type=field_name)
        processed_rois[field_name] = processed
        logger.debug(f"Processed ROI '{field_name}': {processed.shape}")

    return CardROIs(
        full_card=card,
        name_ar=processed_rois.get("name_ar"),
        name_en=processed_rois.get("name_en"),
        address=processed_rois.get("address"),
        id_number=processed_rois.get("id_number"),
        birth_date=processed_rois.get("birth_date"),
        gender=processed_rois.get("gender"),
        nationality=processed_rois.get("nationality"),
        face=processed_rois.get("face"),
        quality_score=quality["overall"]
    )


# ─────────────────────────────────────────────────────────
# Utilities: Debug Visualization
# ─────────────────────────────────────────────────────────

def visualize_rois(card_image: np.ndarray, side: str = "front") -> np.ndarray:
    """
    ارسم مستطيلات على الـ ROIs للـ debugging
    احفظ النتيجة وافتحها للتحقق من صحة الإحداثيات
    """
    vis = card_image.copy()
    h, w = vis.shape[:2]
    rois_map = FRONT_ROIS if side == "front" else BACK_ROIS

    colors = {
        "face": (255, 0, 0),           # أزرق
        "name_ar": (0, 255, 0),        # أخضر
        "name_en": (0, 200, 0),        # أخضر فاتح
        "id_number": (0, 0, 255),      # أحمر
        "birth_date": (255, 165, 0),   # برتقالي
        "gender": (128, 0, 128),       # بنفسجي
        "nationality": (0, 165, 255),  # برتقالي فاتح
        "address": (255, 255, 0),      # أصفر
    }

    for field, (xr, yr, wr, hr) in rois_map.items():
        x = int(xr * w)
        y = int(yr * h)
        x2 = x + int(wr * w)
        y2 = y + int(hr * h)
        color = colors.get(field, (200, 200, 200))
        cv2.rectangle(vis, (x, y), (x2, y2), color, 2)
        cv2.putText(vis, field, (x + 3, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return vis


def save_debug_rois(card_image: np.ndarray, output_dir: str = "debug/") -> None:
    """احفظ كل ROI كصورة منفصلة للـ debugging"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    rois = extract_all_rois(card_image, "front")
    for field, roi in rois.items():
        path = os.path.join(output_dir, f"{field}.png")
        cv2.imwrite(path, roi)
        logger.debug(f"Saved ROI: {path}")

    # احفظ الصورة الكاملة مع الـ bounding boxes
    vis = visualize_rois(card_image, "front")
    cv2.imwrite(os.path.join(output_dir, "_all_rois.png"), vis)
    logger.info(f"Debug images saved to {output_dir}")
```


***

## استخدامه في الـ Pipeline

```python
from app.utils.image_utils import full_preprocess_pipeline, decode_image

# في الـ service
image_bytes = await file.read()
image = decode_image(image_bytes)

# Mode 1: بدون YOLO (coordinate-based)
rois = full_preprocess_pipeline(image, use_yolo=False)

# Mode 2: مع YOLO (أعلى دقة)
yolo_fields = detector.crop_fields(image)  # من detector.py
rois = full_preprocess_pipeline(image, use_yolo=True, yolo_fields=yolo_fields)

# استخدام الـ ROIs
print(rois.quality_score)   # 0.78
print(rois.id_number.shape) # (64, 480)

# للـ debugging — اكتشف مكان الـ ROIs على البطاقة
save_debug_rois(rois.full_card, output_dir="debug/")
```


***

## ترتيب خطوات المعالجة

| الخطوة | الدالة | الهدف |
| :-- | :-- | :-- |
| Quality Check | `assess_quality()` | رفض الصور السيئة مبكراً |
| Resize | `resize_to_standard()` | توحيد الأبعاد |
| Denoise | `remove_noise()` | bilateral filter يحافظ على الحواف [^6_2] |
| Contrast | `enhance_contrast()` | CLAHE على LAB colorspace |
| Perspective | `auto_detect_and_warp()` | تصحيح زاوية التصوير [^6_3] |
| Deskew | `deskew()` | تصحيح الانحراف البسيط [^6_1] |
| Side Detection | `detect_card_side()` | أمامي أم خلفي |
| ROI Extract | `extract_all_rois()` | استخراج كل حقل |
| Line Removal | `remove_background_lines()` | إزالة الخطوط الخلفية [^6_1] |
| Field Preprocess | `preprocess_text_field()` | adaptive threshold لكل نوع حقل [^6_2] |

<span style="display:none">[^6_10][^6_11][^6_12][^6_13][^6_14][^6_15][^6_4][^6_5][^6_6][^6_7][^6_8][^6_9]</span>

<div align="center">⁂</div>

[^6_1]: https://ejle.journals.ekb.eg/article_59924_98b54f20a36e88da1c58ca6fb99a1300.pdf

[^6_2]: https://www.linkedin.com/posts/ramzy-kemmoun-1a3725237_arabicocr-deeplearning-imageprocessing-activity-7397377919339110402-B0BO

[^6_3]: https://www.youtube.com/watch?v=ucZwTgKMC90

[^6_4]: https://github.com/Mostafa-Emad77/Egyptian-ID-Extraction

[^6_5]: https://www.kaggle.com/code/engsaiedali/ocr-egyptian-id-card

[^6_6]: https://stackoverflow.com/questions/24246807/detecting-the-national-id-card-and-getting-the-details

[^6_7]: https://pdfs.semanticscholar.org/657c/854e1f8f80282154a83c35fabb73f38158cc.pdf

[^6_8]: https://pyimagesearch.com/2021/12/01/ocr-passports-with-opencv-and-tesseract/

[^6_9]: https://www.facebook.com/groups/595424764221375/posts/1480570175706825/

[^6_10]: https://kby-ai.com/4-real-life-id-document-ocr-challenges-in-processing/

[^6_11]: https://opencv.org

[^6_12]: https://regulaforensics.com/blog/arabic-script-id-documents-processing/

[^6_13]: https://www.youtube.com/watch?v=SQ3D1tlCtNg

[^6_14]: https://github.com/AHR-OCR2024/Arabic-Handwriting-Recognition

[^6_15]: https://stackoverflow.com/questions/72418797/perspective-transform-omr-sheet-opencv-python


---

# ايه اللي فاضل

بناءً على كل اللي اتبنى لحد دلوقتي، ده اللي فاضل:

***

## اللي اتعمل ✅

- `app/core/config.py` + `logger.py`
- `app/models/detector.py` — ONNX YOLO
- `app/models/ocr_engine.py` — PaddleOCR + EasyOCR
- `app/models/id_parser.py` — تحليل الرقم القومي
- `app/utils/image_utils.py` — Preprocessing + ROI extraction كامل
- `app/utils/text_utils.py` — تنظيف النصوص
- `app/services/pipeline.py` — الـ Orchestrator
- `app/api/routes.py` + `app/main.py` — FastAPI
- `scripts/export_yolo_onnx.py` + `download_models.py`
- `tests/test_pipeline.py`
- `Dockerfile`

***

## اللي فاضل ❌

### 1. `requirements.txt` — لسه مش متكتبش كامل

### 2. `.env` + `.env.example` — إعدادات البيئة

### 3. `scripts/calibrate_rois.py` — أهم حاجة فعلياً

أداة بتعرض البطاقة وتخليك تضبط الـ FRONT_ROIS coordinates بدقة على صورك الحقيقية — لأن الإحداثيات النسبية الحالية تقريبية وممكن تحتاج fine-tuning.

### 4. `app/services/fraud_detection.py` — اختياري لكن مهم

كشف التلاعب في البطاقة (تحقق من consistency الرقم القومي مع باقي الحقول).

### 5. Integration Test حقيقي

اختبار end-to-end بصور بطاقات فعلية أو mock images.

### 6. `README.md` — توثيق التشغيل


***

## الأولوية

```
الأهم الأول:
1. requirements.txt          ← المشروع مش هيشتغل بدونه
2. calibrate_rois.py         ← لازم تضبط الإحداثيات على بطاقاتك
3. .env                      ← settings

اختياري بعدين:
4. fraud_detection.py
5. README.md
```


***

تقولي تبدأ بأيهم؟

---

## `requirements.txt`

```txt
# ─── Web Framework ───────────────────────────────────────
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-multipart==0.0.20

# ─── AI / ML ─────────────────────────────────────────────
onnxruntime==1.21.0           # CPU inference for YOLO
ultralytics==8.3.60           # YOLO export فقط (مش للـ inference)
paddlepaddle==2.6.2           # PaddleOCR backend
paddleocr==2.9.1              # Arabic OCR
easyocr==1.7.2                # Digits + English OCR
torch==2.5.1+cpu              # CPU-only PyTorch (أخف بكتير من GPU version)
torchvision==0.20.1+cpu

# ─── Image Processing ────────────────────────────────────
opencv-python-headless==4.11.0.86   # بدون GUI dependencies
Pillow==11.1.0
numpy==1.26.4
scipy==1.15.1                        # deskew calculations

# ─── Data & Validation ───────────────────────────────────
pydantic==2.10.6
pydantic-settings==2.7.1

# ─── Logging & Utils ─────────────────────────────────────
loguru==0.7.3
python-dotenv==1.0.1

# ─── Testing ─────────────────────────────────────────────
pytest==8.3.4
pytest-asyncio==0.25.3
httpx==0.28.1                        # async test client لـ FastAPI
```

```bash
# تثبيت PyTorch CPU بشكل صريح الأول (عشان pip ميجبش GPU version)
pip install torch==2.5.1+cpu torchvision==0.20.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# بعدين باقي الـ requirements
pip install -r requirements.txt
```


***

## `.env.example`

```env
# ─── App Settings ────────────────────────────────────────
APP_ENV=development                  # development | production
APP_HOST=0.0.0.0
APP_PORT=8000
APP_WORKERS=2                        # عدد الـ uvicorn workers

# ─── Model Paths ─────────────────────────────────────────
YOLO_CARD_MODEL=weights/yolo_card_detect.onnx
YOLO_FIELDS_MODEL=weights/yolo_fields_detect.onnx
MODELS_CACHE_DIR=./models_cache

# ─── YOLO Inference ──────────────────────────────────────
YOLO_CONF_THRESHOLD=0.50
YOLO_IOU_THRESHOLD=0.45
YOLO_INPUT_SIZE=640

# ─── OCR Settings ────────────────────────────────────────
OCR_CPU_THREADS=4
OCR_ENABLE_MKL=true                  # Intel MKL-DNN acceleration

# ─── Card Field Class IDs ────────────────────────────────
# (حسب الـ YOLO training — غيّرها لو الـ classes عندك مختلفة)
CLASS_ID_CARD=0
# CLASS_NAMES بتتحدد في config.py مباشرة

# ─── Image Validation ────────────────────────────────────
MAX_IMAGE_SIZE_MB=10
MIN_QUALITY_SCORE=0.35               # أقل من كده يرفض الصورة
TARGET_IMAGE_WIDTH=1200              # resize target

# ─── API Security (اختياري) ──────────────────────────────
# API_KEY=your_secret_key_here
# RATE_LIMIT_PER_MINUTE=60

# ─── Logging ─────────────────────────────────────────────
LOG_LEVEL=INFO                       # DEBUG | INFO | WARNING | ERROR
LOG_FILE=logs/app.log
LOG_ROTATION=10 MB
LOG_RETENTION=7 days
```


***

## `.env` (للـ development — نفس الـ example بقيم حقيقية)

```env
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
APP_WORKERS=1

YOLO_CARD_MODEL=weights/yolo_card_detect.onnx
YOLO_FIELDS_MODEL=weights/yolo_fields_detect.onnx
MODELS_CACHE_DIR=./models_cache

YOLO_CONF_THRESHOLD=0.50
YOLO_IOU_THRESHOLD=0.45
YOLO_INPUT_SIZE=640

OCR_CPU_THREADS=4
OCR_ENABLE_MKL=true

CLASS_ID_CARD=0

MAX_IMAGE_SIZE_MB=10
MIN_QUALITY_SCORE=0.35
TARGET_IMAGE_WIDTH=1200

LOG_LEVEL=DEBUG
LOG_FILE=logs/app.log
LOG_ROTATION=10 MB
LOG_RETENTION=7 days
```


***

## تحديث `app/core/config.py` ليقرأ الـ `.env`

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_WORKERS: int = 2

    # Model Paths
    YOLO_CARD_MODEL: str = "weights/yolo_card_detect.onnx"
    YOLO_FIELDS_MODEL: str = "weights/yolo_fields_detect.onnx"
    MODELS_CACHE_DIR: str = "./models_cache"

    # YOLO
    YOLO_CONF_THRESHOLD: float = 0.50
    YOLO_IOU_THRESHOLD: float = 0.45
    YOLO_INPUT_SIZE: int = 640

    # OCR
    OCR_CPU_THREADS: int = 4
    OCR_ENABLE_MKL: bool = True

    # Card Fields
    CLASS_ID_CARD: int = 0
    CLASS_NAMES: dict = {
        1: "name_ar", 2: "name_en", 3: "address",
        4: "id_number", 5: "birth_date",
        6: "gender",   7: "nationality"
    }

    # Image Validation
    MAX_IMAGE_SIZE_MB: int = 10
    MIN_QUALITY_SCORE: float = 0.35
    TARGET_IMAGE_WIDTH: int = 1200

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "7 days"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```


***

## `.gitignore` (متنساش تضيفه)

```gitignore
# Environment
.env
venv/
__pycache__/
*.pyc

# Models & Weights (كبيرة جداً للـ git)
weights/*.pt
weights/*.onnx
weights/*.engine
models_cache/

# Logs & Debug
logs/
debug/
*.log

# Test outputs
tests/output/

# IDE
.vscode/
.idea/
```


***

## ملخص الملفات

| الملف | الغرض |
| :-- | :-- |
| `requirements.txt` | كل الـ dependencies مع الـ versions المثبتة |
| `.env` | القيم الفعلية للـ development |
| `.env.example` | template للـ team (يتـ commit على git) |
| `.gitignore` | حماية الـ secrets والموديلات |

> **مهم:** `.env` دايماً في الـ `.gitignore` — بس `.env.example` بيتـ commit عشان الـ team يعرف المتغيرات المطلوبة.

---

# إليك deep research شاملة على أفضل الموديلات الجاهزة من مشاريع فعلية لمشروع Egyptian ID OCR:

🔍 أولاً: موديلات الـ Detection (YOLO)
NASO7Y — Egyptian ID YOLOv8
الريبو الوحيد المتاح مدرّب خصيصًا على البطاقة المصرية. بيشمل Card Detection + Field Detection، ومدعوم بـ Fraud Detection System، وفيه demo حي على [egyptianidocr.streamlit.app](https://egyptianidocr.streamlit.app/).github+1
bash
git clone [https://github.com/NASO7Y/ocr_egyptian_ID.git](https://github.com/NASO7Y/ocr_egyptian_ID.git)

Roboflow — Egyptian National ID Datasets
في workspace من Roboflow بيحتوي على 393 صورة (Egyptian National ID) مع pretrained model وAPI جاهزة. وفي workspace ثاني من يوسف بـ 421 صورة. تقدر تنزلهم بـ YOLOv8 format مباشرةً أو تستخدم الـ pretrained model API.universe.roboflow+2

🔤 ثانياً: موديلات الـ Arabic OCR
الموديلCERWERBLEUالأفضل لـ
QARI v0.2
0.061
0.160
0.737
دقة النص الخام [[arxiv](https://arxiv.org/html/2506.02295v1)]​
QARI v0.3
0.300
0.485
0.545
هيكل المستند + Markdown [[huggingface](https://huggingface.co/NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct)]​
Arabic-English-handwritten v3
1.78%
—
—
الخط المكتوب بخط اليد [[huggingface](https://huggingface.co/sherif1313/Arabic-English-handwritten-OCR-v3)]​
QARI v0.2 هو أفضل open-source Arabic OCR موجود حالياً — بيتفوق على Tesseract بـ 84% في WER ويتفوق على الـ base model بـ 95%.arxiv+2
QARI v0.3 مبني على Qwen2-VL-2B-Instruct وأفضل في فهم هيكل الصفحة والـ HTML/Markdown، لكن دقته الخام أقل من v0.2.[[huggingface](https://huggingface.co/NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct)]​
python

# تحميل QARI v0.2 (الأدق للنص)

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
model = Qwen2VLForConditionalGeneration.from_pretrained(
"NAMAA-Space/Qari-OCR-v0.2-VL-2B-Instruct",
torch_dtype="auto", device_map="auto"
)

Arabic-English Handwritten OCR v3
مبني على Qwen2.5-VL-3B-Instruct ومدرّب على 47,842 sample. بيحقق CER 1.78% على الخط المكتوب يدوياً، أفضل من Google Vision بـ 57%. مفيد لو البطاقات فيها حقول مكتوبة بخط اليد.[[huggingface](https://huggingface.co/sherif1313/Arabic-English-handwritten-OCR-v3)]​

🗂️ ثالثاً: الـ Combination الأمثل لمشروعك
المرحلةالموديلالمصدرالدقة
Card Detection
NASO7Y YOLOv8
GitHub
~95%
Field Detection
NASO7Y YOLOv8
GitHub
~95%
Arabic OCR (نص مطبوع)
QARI v0.2
HuggingFace
CER 0.061
Arabic OCR (نص مكتوب يدوياً)
Arabic-handwritten v3
HuggingFace
CER 1.78%
أرقام/إنجليزي
EasyOCR
pip
عالي

⚠️ ملاحظة مهمة على QARI للـ CPU
QARI قائم على Qwen2-VL (2B parameter) — بيحتاج على الأقل 8GB RAM للـ 8-bit quantization، وبيشتغل على CPU لكن بطيء (~15-30 ثانية للصورة). لو الأداء على CPU مهم جداً، استخدم PaddleOCR كـ fallback أسرع، وQARI للتحقق فقط.[[model.aibase](https://model.aibase.com/models/details/1932644558223708160)]​

دلوقتي عندنا كل التفاصيل الدقيقة — هنكتب `ocr_engine.py` جديد بالكامل مبني على الـ research الحقيقية.

***

## `app/models/ocr_engine.py` — الكود الكامل المحدّث

```python
"""
OCR Engine — Egyptian ID Card
Engines:
  1. QARI v0.2.2.1  → Arabic printed text     (CER: 0.059, WER: 0.221) [web:84]
  2. Handwritten v3 → Arabic handwritten text  (CER: 1.78%)             [web:85]
  3. PaddleOCR      → Fast CPU fallback        (~21ms/field)            [web:60]
  4. EasyOCR        → Digits + English only
"""

import os
import time
import numpy as np
import torch
from PIL import Image
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from app.core.config import settings
from app.core.logger import logger


# ─────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────

class OCRMode(str, Enum):
    QARI      = "qari"        # Arabic printed — أعلى دقة
    HANDWRITE = "handwrite"   # Arabic handwritten
    PADDLE    = "paddle"      # Fast CPU fallback
    EASYOCR   = "easyocr"    # Digits & English


@dataclass
class OCRResult:
    text: str
    confidence: float
    engine_used: OCRMode
    latency_ms: int


# ─────────────────────────────────────────────────────────
# Field → Engine Routing Map
# ─────────────────────────────────────────────────────────

# كل field بيتوجه للـ engine الأنسب له
FIELD_ENGINE_MAP = {
    "name_ar":     OCRMode.QARI,
    "name_en":     OCRMode.EASYOCR,
    "address":     OCRMode.QARI,
    "id_number":   OCRMode.EASYOCR,
    "birth_date":  OCRMode.EASYOCR,
    "gender":      OCRMode.PADDLE,      # كلمة واحدة بسيطة → paddle أسرع
    "nationality": OCRMode.PADDLE,
    "job":         OCRMode.QARI,
    "release_date":OCRMode.EASYOCR,
    "expiry_date": OCRMode.EASYOCR,
}

# الـ QARI prompt المثبتة من الـ paper [web:84]
QARI_PROMPT = (
    "Below is the image of one page of a document, as well as some raw textual "
    "content that was previously extracted for it. Just return the plain text "
    "representation of this document as if you were reading it naturally. "
    "Do not hallucinate."
)


# ─────────────────────────────────────────────────────────
# Engine 1: QARI v0.2.2.1 — Best Arabic Printed OCR
# ─────────────────────────────────────────────────────────

class QariEngine:
    """
    QARI v0.2.2.1 — أفضل Arabic OCR open-source حالياً
    CER: 0.059 | WER: 0.221 | BLEU: 0.597
    مبني على Qwen2-VL-2B-Instruct، متدرّب على 50,000 record [web:84]
    """

    MODEL_NAME = "NAMAA-Space/Qari-OCR-0.2.2.1-Arabic-2B-Instruct"

    def __init__(self, use_4bit: bool = True):
        """
        use_4bit=True → يقلل الـ RAM من ~8GB لـ ~3GB — ضروري للـ CPU [web:80]
        """
        logger.info(f"Loading QARI v0.2.2.1 (4bit={use_4bit})...")
        t0 = time.time()

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
            from qwen_vl_utils import process_vision_info

            self._process_vision_info = process_vision_info

            # 4-bit quantization لتقليل الذاكرة على CPU
            if use_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float32,  # CPU يحتاج float32
                    bnb_4bit_use_double_quant=True,
                )
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.MODEL_NAME,
                    quantization_config=quant_config,
                    device_map="cpu",
                )
            else:
                # بدون quantization — أسرع لكن يحتاج ~8GB RAM
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.MODEL_NAME,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                )

            self.processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
            self.model.eval()

            logger.info(f"✅ QARI loaded in {time.time()-t0:.1f}s")

        except ImportError as e:
            raise ImportError(
                f"QARI dependencies missing: {e}\n"
                "Run: pip install transformers qwen_vl_utils accelerate bitsandbytes"
            )

    def run(self, image_np: np.ndarray, max_tokens: int = 200) -> OCRResult:
        t0 = time.time()

        # تحويل numpy → PIL [web:84]
        image_pil = Image.fromarray(
            cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            if len(image_np.shape) == 3 else image_np
        ).convert("RGB")

        # حفظ مؤقت (Qwen2-VL يحتاج file path) [web:84]
        tmp_path = "/tmp/_qari_input.png"
        image_pil.save(tmp_path)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{tmp_path}"},
                {"type": "text",  "text": QARI_PROMPT},
            ],
        }]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)

        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cpu")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,      # greedy decoding → أسرع + أكثر ثباتاً
            )

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[^9_0].strip()

        # إزالة الملف المؤقت
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        latency = int((time.time() - t0) * 1000)
        logger.debug(f"QARI: '{text[:40]}...' | {latency}ms")

        return OCRResult(
            text=text,
            confidence=0.94,   # QARI لا يرجع confidence score — نستخدم الـ benchmark CER
            engine_used=OCRMode.QARI,
            latency_ms=latency
        )


# ─────────────────────────────────────────────────────────
# Engine 2: Arabic Handwritten OCR v3
# ─────────────────────────────────────────────────────────

class HandwrittenOCREngine:
    """
    Arabic-English-handwritten-OCR-v3
    CER: 1.78% | مبني على Qwen2.5-VL-3B-Instruct
    متدرّب على 47,842 sample — يتفوق على Google Vision بـ 57% [web:85]
    """

    MODEL_NAME = "sherif1313/Arabic-English-handwritten-OCR-v3"

    def __init__(self):
        logger.info("Loading Handwritten OCR v3...")
        t0 = time.time()

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info

            self._process_vision_info = process_vision_info

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.MODEL_NAME,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_NAME,
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
            )
            self.model.eval()
            logger.info(f"✅ Handwritten OCR loaded in {time.time()-t0:.1f}s")

        except ImportError as e:
            raise ImportError(f"Handwritten OCR deps missing: {e}")

    def run(self, image_np: np.ndarray) -> OCRResult:
        t0 = time.time()

        image_pil = Image.fromarray(
            cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            if len(image_np.shape) == 3 else image_np
        ).convert("RGB")

        # Auto resolution reduction — الموديل بيعملها تلقائياً [web:85]
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text",  "text": "اقرأ النص الموجود في هذه الصورة بدقة."},
            ],
        }]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = self._process_vision_info(messages)

        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cpu")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True
        )[^9_0].strip()

        latency = int((time.time() - t0) * 1000)
        return OCRResult(
            text=text,
            confidence=0.982,  # من الـ benchmark: CER 1.78% [web:85]
            engine_used=OCRMode.HANDWRITE,
            latency_ms=latency
        )


# ─────────────────────────────────────────────────────────
# Engine 3: PaddleOCR — Fast CPU Fallback
# ─────────────────────────────────────────────────────────

class PaddleEngine:
    """
    PaddleOCR PP-OCRv4 — أسرع engine على CPU
    ~21ms per field مع MKL-DNN على Intel [web:60]
    """

    def __init__(self):
        logger.info("Loading PaddleOCR...")
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="arabic",
            use_gpu=False,
            cpu_threads=settings.OCR_CPU_THREADS,
            det_model_dir=f"{settings.MODELS_CACHE_DIR}/paddle_det",
            rec_model_dir=f"{settings.MODELS_CACHE_DIR}/paddle_rec",
            cls_model_dir=f"{settings.MODELS_CACHE_DIR}/paddle_cls",
            enable_mkldnn=settings.OCR_ENABLE_MKL,
            show_log=False,
        )
        logger.info("✅ PaddleOCR ready")

    def run(self, image_np: np.ndarray) -> OCRResult:
        t0 = time.time()
        try:
            result = self.ocr.ocr(image_np, cls=True)
            if not result or not result[^9_0]:
                return OCRResult("", 0.0, OCRMode.PADDLE, int((time.time()-t0)*1000))
            texts = [line[^9_1][^9_0] for line in result[^9_0] if line]
            confs = [line[^9_1][^9_1] for line in result[^9_0] if line]
            return OCRResult(
                text=" ".join(texts),
                confidence=float(np.mean(confs)) if confs else 0.0,
                engine_used=OCRMode.PADDLE,
                latency_ms=int((time.time()-t0)*1000)
            )
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return OCRResult("", 0.0, OCRMode.PADDLE, 0)


# ─────────────────────────────────────────────────────────
# Engine 4: EasyOCR — Digits & English
# ─────────────────────────────────────────────────────────

class EasyOCREngine:
    """
    EasyOCR — للأرقام والإنجليزي فقط
    أدق وأسرع من الـ VLMs للأرقام اللاتينية
    """

    def __init__(self):
        logger.info("Loading EasyOCR...")
        import easyocr
        self.reader = easyocr.Reader(
            ['ar', 'en'],
            gpu=False,
            quantize=True,
            model_storage_directory=f"{settings.MODELS_CACHE_DIR}/easyocr",
            verbose=False,
        )
        logger.info("✅ EasyOCR ready")

    def run(self, image_np: np.ndarray, digits_only: bool = False) -> OCRResult:
        t0 = time.time()
        try:
            kwargs = dict(
                detail=1,
                decoder="greedy",
                beamWidth=1,
            )
            if digits_only:
                kwargs["allowlist"] = "0123456789"

            results = self.reader.readtext(image_np, **kwargs)
            if not results:
                return OCRResult("", 0.0, OCRMode.EASYOCR, int((time.time()-t0)*1000))

            text = " ".join([r[^9_1] for r in results])
            conf = float(np.mean([r[^9_2] for r in results]))
            return OCRResult(
                text=text,
                confidence=conf,
                engine_used=OCRMode.EASYOCR,
                latency_ms=int((time.time()-t0)*1000)
            )
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return OCRResult("", 0.0, OCRMode.EASYOCR, 0)


# ─────────────────────────────────────────────────────────
# Main OCR Manager — Smart Routing + Fallback
# ─────────────────────────────────────────────────────────

import cv2

class OCREngine:
    """
    Smart OCR Router:
    - يوجه كل field للـ engine الأنسب
    - لو QARI بطيء على جهازك → fallback تلقائي لـ PaddleOCR
    - لو الـ confidence منخفض → يجرب engine ثاني
    """

    _instance = None

    # Confidence threshold — لو أقل منه بيجرب fallback
    CONFIDENCE_THRESHOLD = 0.60
    # Time threshold — لو QARI أخد أكتر من كده بيـ fallback (ثانية)
    QARI_TIMEOUT_WARN_S = 45

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(
        self,
        enable_qari: bool = True,
        enable_handwritten: bool = False,   # False بالـ default — ثقيل جداً
        qari_4bit: bool = True,
    ):
        """
        enable_qari: False → يستخدم PaddleOCR فقط (أسرع، أقل دقة)
        enable_handwritten: True → لو البطاقة مكتوبة بخط اليد
        qari_4bit: True → يقلل RAM من 8GB → 3GB [web:84]
        """
        if self._initialized:
            logger.info("OCR Engine already initialized, skipping")
            return

        logger.info("=" * 50)
        logger.info("Initializing OCR Engine...")

        self.enable_qari = enable_qari
        self.enable_handwritten = enable_handwritten

        # دايماً محتاجين Paddle + Easy
        self.paddle = PaddleEngine()
        self.easy   = EasyOCREngine()

        # QARI اختياري (حسب الـ RAM المتاح)
        if enable_qari:
            try:
                self.qari = QariEngine(use_4bit=qari_4bit)
            except Exception as e:
                logger.warning(f"QARI failed to load ({e}), falling back to PaddleOCR")
                self.qari = None
                self.enable_qari = False
        else:
            self.qari = None

        # Handwritten OCR اختياري
        if enable_handwritten:
            try:
                self.handwritten = HandwrittenOCREngine()
            except Exception as e:
                logger.warning(f"Handwritten OCR failed ({e})")
                self.handwritten = None
                self.enable_handwritten = False
        else:
            self.handwritten = None

        self._initialized = True
        logger.info("✅ OCR Engine initialized")
        logger.info("=" * 50)

    def _route(self, field_name: str) -> OCRMode:
        """تحديد أي engine يستخدم لكل field"""
        mode = FIELD_ENGINE_MAP.get(field_name, OCRMode.QARI)

        # لو QARI مش متاح → رجوع لـ Paddle
        if mode == OCRMode.QARI and not self.enable_qari:
            return OCRMode.PADDLE

        # لو Handwritten مش متاح → رجوع لـ QARI
        if mode == OCRMode.HANDWRITE and not self.enable_handwritten:
            return OCRMode.QARI if self.enable_qari else OCRMode.PADDLE

        return mode

    def _run_engine(
        self, mode: OCRMode, image_np: np.ndarray,
        digits_only: bool = False
    ) -> OCRResult:
        """تشغيل engine محدد"""
        if mode == OCRMode.QARI and self.qari:
            return self.qari.run(image_np)
        elif mode == OCRMode.HANDWRITE and self.handwritten:
            return self.handwritten.run(image_np)
        elif mode == OCRMode.EASYOCR:
            return self.easy.run(image_np, digits_only=digits_only)
        else:
            return self.paddle.run(image_np)

    def ocr_field(
        self,
        image_np: np.ndarray,
        field_name: str,
        is_handwritten: bool = False,
        enable_fallback: bool = True,
    ) -> OCRResult:
        """
        Main entry point — OCR لـ field واحد مع smart routing + fallback

        Args:
            image_np:       صورة الـ field المعالجة
            field_name:     اسم الـ field (من FIELD_ENGINE_MAP)
            is_handwritten: True لو الكتابة بخط اليد
            enable_fallback: True → يجرب engine ثاني لو الـ confidence منخفض
        """
        if not self._initialized:
            raise RuntimeError("OCR Engine not initialized. Call .initialize() first")

        # تحديد الـ engine
        mode = OCRMode.HANDWRITE if is_handwritten else self._route(field_name)
        digits_only = field_name in {"id_number", "birth_date", "release_date", "expiry_date"}

        # تشغيل الـ primary engine
        result = self._run_engine(mode, image_np, digits_only=digits_only)

        logger.info(
            f"[{field_name}] {result.engine_used.value} → "
            f"'{result.text[:30]}' | conf={result.confidence:.2f} | {result.latency_ms}ms"
        )

        # Fallback لو الـ confidence منخفض
        if enable_fallback and result.confidence < self.CONFIDENCE_THRESHOLD:
            fallback_mode = OCRMode.PADDLE if mode != OCRMode.PADDLE else OCRMode.EASYOCR
            logger.warning(
                f"[{field_name}] Low confidence ({result.confidence:.2f}), "
                f"trying fallback: {fallback_mode.value}"
            )
            fallback_result = self._run_engine(fallback_mode, image_np, digits_only=digits_only)

            # اختار الأعلى confidence
            if fallback_result.confidence > result.confidence:
                logger.info(f"[{field_name}] Fallback won: {fallback_result.confidence:.2f}")
                return fallback_result

        return result

    def ocr_all_fields(
        self,
        fields: dict[str, np.ndarray],
        is_handwritten: bool = False,
    ) -> dict[str, OCRResult]:
        """OCR لكل الـ fields دفعة واحدة"""
        return {
            field_name: self.ocr_field(field_img, field_name, is_handwritten)
            for field_name, field_img in fields.items()
            if field_img is not None and field_img.size > 0
        }
```


***

## تحديث `app/main.py` — إعدادات التشغيل

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.services.pipeline import IDExtractionPipeline
from app.models.ocr_engine import OCREngine
from app.core.logger import logger
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # إعداد OCR Engine حسب الـ RAM المتاح
    ram_gb = int(os.getenv("AVAILABLE_RAM_GB", "4"))

    ocr = OCREngine()
    ocr.initialize(
        enable_qari=ram_gb >= 4,          # QARI يحتاج 3GB+ مع 4bit
        enable_handwritten=ram_gb >= 8,   # Handwritten يحتاج 6GB+
        qari_4bit=ram_gb < 8,             # 4bit لو RAM أقل من 8GB
    )

    IDExtractionPipeline.initialize()
    logger.info(f"🚀 App ready | RAM mode: {ram_gb}GB | QARI: {ocr.enable_qari}")
    yield

app = FastAPI(
    title="Egyptian ID OCR API",
    version="2.0.0",
    lifespan=lifespan
)

app.include_router(router)
```


***

## تحديث `requirements.txt` — Dependencies الإضافية

```txt
# أضف دول للـ requirements.txt الموجود
qwen_vl_utils==0.0.8          # Qwen2-VL image processing [web:84]
accelerate>=0.26.0             # model loading optimization
bitsandbytes==0.45.0           # 4-bit quantization للـ CPU
```


***

## Performance على CPU — التوقعات الواقعية

| الـ Engine | الـ Field | الوقت على CPU | الدقة |
| :-- | :-- | :-- | :-- |
| **QARI v0.2.2.1** | name_ar, address | 15–30 ثانية | CER 0.059 [^9_1] |
| **Handwritten v3** | حقول يدوية | 20–40 ثانية | CER 1.78% [^9_2] |
| **PaddleOCR** | gender, nationality | ~21ms | جيدة |
| **EasyOCR** | id_number, dates | ~200ms | عالية للأرقام |

> ⚡ **الاستراتيجية:** QARI للدقة القصوى على الحقول المهمة (الاسم والعنوان)، وPaddleOCR للحقول البسيطة — ده بيقلل الوقت الكلي من 3 دقائق لـ ~45 ثانية على CPU عادي.[^9_3][^9_1]
<span style="display:none">[^9_10][^9_11][^9_12][^9_13][^9_14][^9_15][^9_4][^9_5][^9_6][^9_7][^9_8][^9_9]</span>

<div align="center">⁂</div>

[^9_1]: https://model.aibase.com/models/details/1915718612784144385

[^9_2]: https://huggingface.co/sherif1313/Arabic-English-handwritten-OCR-v3

[^9_3]: https://huggingface.co/NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct

[^9_4]: https://huggingface.co/NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct

[^9_5]: https://huggingface.co/NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct

[^9_6]: https://arxiv.org/pdf/2506.02295.pdf

[^9_7]: https://arxiv.org/html/2506.02295v1

[^9_8]: https://www.reddit.com/r/Arabic_Osr_Historical/comments/1qh3422/arabicenglishhandwrittenocrqwen3vl4b/

[^9_9]: https://groups.google.com/g/sigarab/c/SV2v6rJO75c

[^9_10]: https://huggingface.co/mradermacher/Arabic-English-handwritten-OCR-v3-GGUF

[^9_11]: https://ae.linkedin.com/posts/mohammad-essam97_%D8%B9%D9%86%D8%AF%D9%83-%D9%85%D8%B4%D9%83%D9%84%D8%A9-%D9%81%D9%89-%D9%86%D9%85%D8%A7%D8%B0%D8%AC-%D8%A7%D9%84-ocr-%D8%A8%D8%A7%D9%84%D9%84%D8%BA%D8%A9-%D8%A7%D9%84%D8%B9%D8%B1%D8%A8%D9%8A%D8%A9-activity-7302650473507905537-lKo5

[^9_12]: https://ocr.space

[^9_13]: https://www.promptlayer.com/models/qari-ocr-0221-vl-2b-instruct

[^9_14]: https://www.arabic-ocr.com

[^9_15]: https://arxiv.org/abs/2506.02295


---

# ده وش البطاقه المصريه

تمام — الصورة اتحللت بنجاح. ده اللي استخرجناه من البطاقة الحقيقية:

***

## نتائج الـ ROI Extraction على بطاقتك

الـ ROIs اتاستخرجت بنجاح من الصورة (555×352 px):


| الـ Field | اللون | المحتوى المرئي |
| :-- | :-- | :-- |
| **face** 🔴 | أحمر | الصورة الشخصية (يسار) |
| **name_ar** 🟢 | أخضر | علياء / محمود احمد نصر شلبى / العنوان |
| **birth_date** 🔵 | أزرق فاتح | `١٩٩٧/١٠/٠٥` |
| **id_number** 🔵 | أزرق غامق | `٢٩٧٠١٠٥٠١٨٤١` |
| **serial** 🟣 | بنفسجي | `GD2905373` |


***

## ملاحظات على الـ Layout من الصورة الحقيقية

بعد ما شفنا البطاقة الفعلية، لازم نعدّل الـ `FRONT_ROIS` في `image_utils.py`:

```python
# إحداثيات مضبوطة على البطاقة المصرية الحقيقية
FRONT_ROIS = {
    "face":       (0.02, 0.07, 0.27, 0.80),   # الصورة الشخصية — يسار

    # الاسم والعنوان في نفس المنطقة (سطرين أو 3)
    "name_ar":    (0.30, 0.07, 0.68, 0.44),   # الاسم + العنوان كتلة واحدة

    # الرقم القومي وتاريخ الميلاد في نفس السطر
    "birth_date": (0.30, 0.62, 0.38, 0.18),   # يمين السطر الأخير
    "id_number":  (0.38, 0.62, 0.62, 0.18),   # وسط السطر الأخير

    # الرقم التسلسلي — أسفل يسار
    "serial":     (0.01, 0.82, 0.28, 0.15),
}
```


***

## ملاحظة مهمة على البطاقة دي

من الصورة واضح إن:

- **الاسم والعنوان** في نفس المنطقة بدون separator واضح — هيحتاج QARI يفصل بينهم
- **الرقم القومي** `٢٩٧٠١٠٥٠١٨٤١` مكتوب بالأرقام العربية (ليس اللاتينية) — محتاج `arabic_to_english_digits()` في الـ post-processing:

```python
def arabic_to_english_digits(text: str) -> str:
    """تحويل الأرقام العربية/الفارسية للإنجليزية"""
    arabic_digits = '٠١٢٣٤٥٦٧٨٩'
    english_digits = '0123456789'
    table = str.maketrans(arabic_digits, english_digits)
    return text.translate(table)

# مثال:
# "٢٩٧٠١٠٥٠١٨٤١" → "29701050184 1"
```

> ده اكتشاف مهم جداً — كتير من الـ OCR engines بترجع الأرقام بالعربي، وده بيكسر `parse_national_id()` لو مش عاملين conversion. أضفها في `text_utils.py` وادعيها قبل الـ validation.
<span style="display:none">[^10_1]</span>

<div align="center">⁂</div>

[^10_1]: Screenshot-2025-12-31-031335_png.rf.d512aad875172bd9c6f3f04a4f059df1.jpg

