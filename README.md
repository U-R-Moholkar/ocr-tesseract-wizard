
---

```markdown
# 📄 OCR-Based Form Parser with LLM JSON Extraction

This project processes scanned or photographed documents to extract clean, structured data using a pipeline of image preprocessing, skew correction, OCR, and LLM-powered parsing.

---

## 🧰 Requirements

- Python 3.7+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- OpenAI-compatible API key (for GPT-4o via OpenRouter or OpenAI)

### 📦 Python Libraries

```bash
pip install -r requirements.txt
```
---

### 🔧 Configuration

#### 🗂️ Creating a `.env` File

Create a `.env` file in the root directory of the project and add the necessary environment variables in the following format:

```env
# Example .env file
OPENAI_API_KEY="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
BASE_DATA_PATH=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
TESSERACT_CMD=C:/xx/xx/tesseract.exe

```
---
---

## 🔄 Workflow Overview

1. 📷 **Load and preprocess scanned image**
2. 📐 **Correct skew**
3. 🎨 **Apply thresholding**
4. 🔍 **Extract text using Tesseract OCR**
5. 🤖 **Use GPT-4o (via LangChain) to convert raw text into structured JSON**
6. 📊 **Parse JSON into DataFrame**

---

## 1️⃣ Image Skew Correction

```python
import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

def correct_skew(image, delta=0.5, limit=15, debug=False):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    angles = np.arange(-limit, limit + delta, delta)
    scores = [determine_score(thresh, angle) for angle in angles]
    best_angle = angles[scores.index(max(scores))]

    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return best_angle, corrected
```

---

## 2️⃣ Thresholding for OCR Readability

```python
gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 10))
plt.imshow(thresh, cmap='gray')
plt.title("Thresholded Image")
plt.axis("off")
plt.show()
```

---

## 3️⃣ OCR with Tesseract

```python
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(gray, config=custom_config)

print("📝 OCR Extracted Text:\n", text)
```

---

## 4️⃣ GPT-4o-Based JSON Extraction (LangChain)

```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

prompt = f"""
You are given raw OCR text from a scanned document. Extract any tabular data into clean JSON.

Return **only valid JSON** — no explanations, no extra text.

Example format:
[
  {{
    "Name": "John Doe",
    "Street Address": "123 Main St",
    "City": "Los Angeles",
    "State": "CA",
    "ZIP": "90001",
    "Telephone": "555-1234",
    "Appointment Date": "2025-01-01"
  }},
  ...
]

Raw OCR Text:
{text}
"""

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model="openai/gpt-4o",
    max_tokens=1500,
    temperature=0.7,
)

messages = [HumanMessage(content=prompt)]
response = llm.invoke(messages)
print("\n🧠 LLM Response (Structured Data):\n", response.content)
```

---

## 5️⃣ Extract JSON and Convert to DataFrame

```python
import re
import json
import pandas as pd

match = re.search(r'\[\s*{.*?}\s*\]', response.content, re.DOTALL)
if match:
    json_str = match.group(0)
    data = json.loads(json_str)
    df = pd.DataFrame(data)
    print(df)
else:
    print("No JSON array found in response.")
```

---

## 📌 Notes

- Make sure Tesseract is installed and the path is correctly set in `pytesseract.pytesseract.tesseract_cmd`.
- GPT-4o is accessed via OpenRouter in this example, but you can configure it for OpenAI directly if needed.
- You can customize the prompt for more structured or domain-specific data extraction.

---

## 📷 Example Input

> A scanned or photographed document (e.g., paper patient intake,appointment document, bank statement, transaction documents)

## 📤 Output

> A JSON list of entries like:

```json
[
  {
    "Name": "Jane Smith",
    "Street Address": "432 Elm St",
    "City": "Anaheim",
    "State": "CA",
    "ZIP": "92801",
    "Telephone": "555-9876",
    "Appointment Date": "2025-04-06"
  }
]

Output file:
output/extracted_data.csv
```

---

## 🚀 Author

Built with ❤️ by Utkarsh R. Moholkar

---
```

