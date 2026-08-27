from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import easyocr
from PIL import Image
import io
import piexif
import requests
import base64
import os
from typing import Optional
import numpy as np

app = FastAPI()

# Enable CORS (so your HF frontend can call this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EasyOCR once at startup (runs on CPU)
reader = easyocr.Reader(['en'], gpu=False)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    groq_api_key: Optional[str] = Form(None)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert PIL image to numpy array for EasyOCR
        img_np = np.array(image)

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

        # ----- 2. OCR with EasyOCR (no system dependencies needed) -----
        result = reader.readtext(img_np)
        ocr_results = []
        for (bbox, text, confidence) in result:
            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            coord_str = f"[{bbox[0][0]:.0f},{bbox[0][1]:.0f}] → [{bbox[1][0]:.0f},{bbox[1][1]:.0f}] → [{bbox[2][0]:.0f},{bbox[2][1]:.0f}] → [{bbox[3][0]:.0f},{bbox[3][1]:.0f}]"
            ocr_results.append({
                "text": text,
                "confidence": round(confidence, 2),
                "coordinates": coord_str
            })

        # ----- 3. AI Reasoning (only if API key provided) -----
        reasoning = "AI reasoning not available (no API key provided)."
        if groq_api_key and groq_api_key.strip():
            try:
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                headers = {
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "llama-3.2-90b-vision-preview",
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
