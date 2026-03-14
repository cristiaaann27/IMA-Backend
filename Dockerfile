## ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Evitar archivos .pyc y buffering en logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instalar dependencias de compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
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

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/deps

# Solo curl para healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root antes de copiar archivos
RUN useradd -m -u 1001 appuser

# Copiar dependencias pre-compiladas desde builder a ruta propia
# Evita mezclar con site-packages del sistema y es más robusto ante cambios de la imagen base
COPY --from=builder /build/deps /app/deps

# Copiar código de la aplicación con el usuario correcto
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/

# Crear directorios necesarios con permisos correctos
# Nota: en producción estos son sobreescritos por los volúmenes de docker-compose
RUN mkdir -p models data/curated data/predictions data/processed data/raw reports \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# 1 worker + preload: comparte el modelo en memoria sin saturar la RAM de t2.micro
# uvicorn maneja concurrencia async internamente, no se necesitan múltiples procesos
CMD ["gunicorn", "app.main:app", \
    "--workers", "1", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8000", \
    "--timeout", "300", \
    "--preload", \
    "--access-logfile", "-", \
    "--error-logfile", "-"]