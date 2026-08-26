from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import easyocr
import piexif
from PIL import Image
import io
import base64
import requests
import os

app = FastAPI()

# Allow your Hugging Face Space to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load EasyOCR (runs perfectly on Render's free 512MB RAM)
reader = easyocr.Reader(['en'], gpu=False)

# Optional: Get free key from open.bigmodel.cn for AI reasoning (skip if you want)
GLM_API_KEY = os.getenv("GLM_API_KEY", "")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 1. EXIF Metadata
        exif_data = {}
        try:
            exif_dict = piexif.load(contents)
            for ifd in exif_dict:
                for tag, value in exif_dict[ifd].items():
                    tag_name = piexif.TAGS[ifd].get(tag, {}).get('name', str(tag))
                    exif_data[tag_name] = str(value)
        except:
            pass

        # 2. OCR – Text + Bounding Boxes
        result = reader.readtext(contents)
        ocr_results = []
        for (bbox, text, confidence) in result:
            coord_str = f"[{bbox[0][0]:.0f},{bbox[0][1]:.0f}] → [{bbox[1][0]:.0f},{bbox[1][1]:.0f}] → [{bbox[2][0]:.0f},{bbox[2][1]:.0f}] → [{bbox[3][0]:.0f},{bbox[3][1]:.0f}]"
            ocr_results.append({
                "text": text,
                "confidence": round(confidence, 2),
                "coordinates": coord_str
            })

        # 3. AI Reasoning (if API key is set)
        reasoning = "AI reasoning not configured. OCR and EXIF extracted successfully."
        if GLM_API_KEY:
            try:
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                headers = {"Authorization": f"Bearer {GLM_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": "glm-4.6v-flash",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail (objects, scene, text seen). Give a short OSINT summary."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        ]
                    }]
                }
                resp = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions", headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    reasoning = resp.json()['choices'][0]['message']['content']
            except Exception as e:
                reasoning = f"AI API error: {str(e)}"

        # 4. Return JSON response
        return {
            "success": True,
            "reasoning": reasoning,
            "metadata": {
                "Camera": exif_data.get("Make", "N/A") + " " + exif_data.get("Model", ""),
                "Lens": exif_data.get("LensModel", "N/A"),
                "Date": exif_data.get("DateTime", "N/A"),
                "GPS": exif_data.get("GPSLatitude", "N/A"),
                "Dimensions": f"{image.width} x {image.height}",
                "Size": f"{len(contents) / 1024:.1f} KB"
            },
            "ocr": ocr_results,
            "reverse_search": [
                {"site": "Google Lens", "url": "https://lens.google.com/"},
                {"site": "TinEye", "url": "https://tineye.com/"}
            ]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/")
def root():
    return {"message": "VisionEyee Backend is alive!"}
