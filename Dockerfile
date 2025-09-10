# Use lightweight Python image
FROM python:3.11-slim

# Install system dependencies needed by YOLO/Ultralytics
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code into container
COPY . .

# Ensure folders exist
RUN mkdir -p /app/models /app/uploads /app/dataset_images /app/static

# Run app on port 8000
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
