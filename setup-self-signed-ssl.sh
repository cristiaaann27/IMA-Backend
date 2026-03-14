#!/bin/bash

# Script para generar certificado autofirmado
# Ejecutar en tu instancia EC2

echo "=== Generando certificado autofirmado ==="

# Detectar IP pública de la instancia EC2
EC2_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || hostname -I | awk '{print $1}')
echo "IP detectada: ${EC2_IP}"

# Crear directorio para certificados
sudo mkdir -p /etc/nginx/ssl

# Generar certificado autofirmado (válido por 1 año)
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/selfsigned.key \
    -out /etc/nginx/ssl/selfsigned.crt \
    -subj "/C=CO/ST=State/L=City/O=IMA/CN=${EC2_IP}"

echo "✅ Certificado autofirmado generado en:"
echo "   - Certificado: /etc/nginx/ssl/selfsigned.crt"
echo "   - Clave: /etc/nginx/ssl/selfsigned.key"
