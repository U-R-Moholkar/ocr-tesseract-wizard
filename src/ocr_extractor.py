# ocr_extractor.py

import os
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()
tess_path = os.getenv("TESSERACT_CMD")
print("Tesseract Path:", tess_path)

pytesseract.pytesseract.tesseract_cmd = tess_path

api_key = os.getenv("OPENAI_API_KEY")


if not pytesseract.pytesseract.tesseract_cmd:
    raise ValueError("TESSERACT_CMD is not set in your .env file.")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in your .env file.")


class OCRToJSONExtractor:
    def __init__(self):
        
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            model="openai/gpt-4o",
            max_tokens=1500,
            temperature=0.7,
        )

    def extract_text_from_image(self, image):
        """Extract raw text from an image using Tesseract OCR."""
        custom_config = r'--oem 3 --psm 6'
        return pytesseract.image_to_string(image, config=custom_config)

    def convert_ocr_text_to_json(self, ocr_text):
        """Send OCR text to LLM and receive structured JSON."""
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
                    {ocr_text}
                    """
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return response.content

    def process_image_to_json(self, image):
        """Complete pipeline: OCR → LLM → JSON"""
        ocr_text = self.extract_text_from_image(image)
        print(f"Raw OCR Text:===========================>\n{ocr_text}")
        return self.convert_ocr_text_to_json(ocr_text)
