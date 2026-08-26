from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pytesseract
from PIL import Image
import io
import piexif
import requests
import base64
import os
from typing import Optional

app = FastAPI()

# Enable CORS (so your HF frontend can call this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API endpoint (free vision model)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    groq_api_key: Optional[str] = Form(None)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # ----- 1. EXIF Metadata -----
        exif_data = {}
        try:
            exif_dict = piexif.load(contents)
            for ifd in exif_dict:
                for tag, value in exif_dict[ifd].items():
                    tag_name = piexif.TAGS[ifd].get(tag, {}).get('name', str(tag))
                    exif_data[tag_name] = str(value)
        except:
            pass

        # ----- 2. OCR with Tesseract -----
        gray = image.convert('L')
        ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

        ocr_results = []
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 30:
                text = ocr_data['text'][i].strip()
                if text:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    coord_str = f"[{x},{y}] → [{x+w},{y}] → [{x+w},{y+h}] → [{x},{y+h}]"
                    ocr_results.append({
                        "text": text,
                        "confidence": round(ocr_data['conf'][i] / 100, 2),
                        "coordinates": coord_str
                    })

        # ----- 3. AI Reasoning (only if API key provided) -----
        reasoning = "AI reasoning not available (no API key provided)."
        if groq_api_key and groq_api_key.strip():
            try:
                # Convert image to base64
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                headers = {
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "llama-3.2-90b-vision-preview",  # Groq's free vision model
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image in detail. What objects, people, text, or landmarks are visible? Give a concise OSINT summary."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                            ]
                        }
                    ],
                    "max_tokens": 300,
                    "temperature": 0.7
                }
                resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    reasoning = resp.json()['choices'][0]['message']['content']
                else:
                    reasoning = f"Groq API error: {resp.status_code}"
            except Exception as e:
                reasoning = f"AI reasoning failed: {str(e)}"

        # ----- 4. Reverse Search (mock) -----
        reverse_results = [
            {"site": "Google Lens", "url": "https://lens.google.com/"},
            {"site": "TinEye", "url": "https://tineye.com/"}
        ]

        return {
            "success": True,
            "reasoning": reasoning,
            "metadata": {
                "Camera": exif_data.get("Make", "N/A") + " " + exif_data.get("Model", ""),
                "Lens": exif_data.get("LensModel", "N/A"),
                "Date": exif_data.get("DateTime", "N/A"),
                "GPS": exif_data.get("GPSLatitude", "N/A"),
                "Altitude": exif_data.get("GPSAltitude", "N/A"),
                "ISO": exif_data.get("ISOSpeedRatings", "N/A"),
                "FocalLength": exif_data.get("FocalLength", "N/A"),
                "Aperture": exif_data.get("FNumber", "N/A"),
                "ShutterSpeed": exif_data.get("ExposureTime", "N/A"),
                "Dimensions": f"{image.width} x {image.height}",
                "Size": f"{len(contents) / 1024:.1f} KB"
            },
            "ocr": ocr_results,
            "reverse_search": reverse_results
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/")
def root():
    return {"message": "VisionEyee Backend is alive!"}
