#!/bin/bash
set -e

echo "=== Despliegue Backend en EC2 ==="

# Detener contenedores existentes
echo "Deteniendo contenedores..."
docker-compose down

# Limpiar imágenes antiguas
echo "Limpiando imágenes antiguas..."
docker system prune -f

# Construir y levantar servicios
echo "Construyendo y levantando servicios..."
docker-compose up -d --build

# Verificar estado
echo "Verificando estado de los servicios..."
docker-compose ps

echo "=== Despliegue completado ==="
echo "API disponible en http://localhost"
echo "Logs: docker-compose logs -f"
