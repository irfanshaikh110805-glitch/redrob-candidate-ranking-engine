FROM python:3.11-slim

# Install compilation dependencies (needed for certain C++ dependencies like FAISS/RapidFuzz if pre-built wheels aren't used)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better Docker caching
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, offline model cache, and output folders
COPY backend/ backend/
COPY .model_cache/ .model_cache/
COPY output/ output/

# Guarantee offline sentence-transformers execution
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8000

# Launch FastAPI app
CMD ["python", "backend/app.py"]
