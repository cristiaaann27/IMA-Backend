#!/bin/bash
set -e

echo "=== Despliegue Backend en EC2 ==="

# Detener contenedores existentes
echo "Deteniendo contenedores..."
docker compose down

# Limpiar imágenes antiguas
echo "Limpiando imágenes antiguas..."
docker system prune -f

# Construir y levantar servicios
echo "Construyendo y levantando servicios..."
docker compose up -d --build

# Verificar estado
echo "Verificando estado de los servicios..."
docker compose ps

echo "=== Despliegue completado ==="
EC2_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || hostname -I | awk '{print $1}')
echo "API disponible en http://${EC2_IP}/api/v1"
echo "Logs: docker compose logs -f"
