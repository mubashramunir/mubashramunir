# Base image
FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app.py utils.py README.md ./

# Env (tweak defaults if needed)
ENV MODEL_NAME=yolov8n.pt \
    FRAME_STRIDE=3 \
    CONF_THRESH=0.25 \
    IOU_THRESH=0.45

# Expose port for Spaces
EXPOSE 7860

# Launch app
CMD ["python", "app.py"]

