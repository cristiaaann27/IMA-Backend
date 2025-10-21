#!/bin/bash

# Script para generar certificado autofirmado
# Ejecutar en tu instancia EC2

echo "=== Generando certificado autofirmado ==="

# Crear directorio para certificados
sudo mkdir -p /etc/nginx/ssl

# Generar certificado autofirmado (válido por 1 año)
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/selfsigned.key \
    -out /etc/nginx/ssl/selfsigned.crt \
    -subj "/C=CO/ST=State/L=City/O=IMA/CN=98.81.248.198"

echo "✅ Certificado autofirmado generado en:"
echo "   - Certificado: /etc/nginx/ssl/selfsigned.crt"
echo "   - Clave: /etc/nginx/ssl/selfsigned.key"
