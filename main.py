from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pytesseract
from PIL import Image, ImageFilter
import io
import piexif
import requests
import os

app = FastAPI()

# CORS for HF frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: GLM API key for AI reasoning (free from open.bigmodel.cn)
GLM_API_KEY = os.getenv("GLM_API_KEY", "")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
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

        # ----- 2. OCR with Tesseract (lightweight) -----
        # Preprocess image for better OCR
        gray = image.convert('L')
        # Use pytesseract to get data with bounding boxes
        ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

        ocr_results = []
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 30:  # confidence threshold
                text = ocr_data['text'][i].strip()
                if text:
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    # Convert to 4-point coordinates (x1,y1 → x2,y2 → x3,y3 → x4,y4)
                    coord_str = f"[{x},{y}] → [{x+w},{y}] → [{x+w},{y+h}] → [{x},{y+h}]"
                    ocr_results.append({
                        "text": text,
                        "confidence": round(ocr_data['conf'][i] / 100, 2),
                        "coordinates": coord_str
                    })

        # ----- 3. AI Reasoning (optional, uses GLM API) -----
        reasoning = "AI reasoning not configured. OCR and EXIF extracted successfully."
        if GLM_API_KEY:
            try:
                import base64
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

        # ----- 4. Response -----
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
