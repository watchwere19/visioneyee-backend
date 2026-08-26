FROM python:3.11-slim

# Create the missing directory and install Tesseract
RUN mkdir -p /var/lib/apt/lists/partial \
    && apt-get update \
    && apt-get install -y tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
