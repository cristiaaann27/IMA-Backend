## ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Instalar dependencias de compilación
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instalar dependencias en directorio aislado (PyTorch CPU-only)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt \
    --target /build/deps

## ---- Stage 2: Runtime ----
FROM python:3.11-slim

WORKDIR /app

# Solo curl para healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias pre-compiladas desde builder
COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages/

# Copiar código de la aplicación
COPY app/ ./app/
COPY src/ ./src/
COPY configs/ ./configs/

# Crear directorios necesarios (estos serán montados como volúmenes en producción)
RUN mkdir -p models data/curated data/predictions data/processed data/raw reports

# Exponer puerto
EXPOSE 8000

# Comando de inicio (2 workers + preload para compartir memoria en Free Tier)
CMD ["gunicorn", "app.main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300", "--preload", "--access-logfile", "-", "--error-logfile", "-"]
