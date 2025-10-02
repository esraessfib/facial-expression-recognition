FROM python:3.10-slim

WORKDIR /app

# Install system deps (for opencv)
RUN apt-get update && apt-get install -y \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app ./app
COPY ./models ./models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
