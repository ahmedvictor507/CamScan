FROM python:3.10-slim

WORKDIR /app

# opencv-python-headless (requirements-deploy.txt) needs libglib2.0-0 but not libgl1/X11
# -- headless builds skip the GUI bindings that would otherwise need those.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY camscan/ camscan/
COPY api/ api/
COPY ["model/attempt 5_yolo26s/weights/best.onnx", "model/attempt 5_yolo26s/weights/best.onnx"]

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
