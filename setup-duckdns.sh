#!/bin/bash
set -e

echo "=== Configuración DuckDNS ==="
echo ""

# ---- Solicitar datos ----
read -p "Subdominio DuckDNS (ej: ima-api): " DUCKDNS_SUBDOMAIN

# -s oculta el token mientras se escribe
read -s -p "Token DuckDNS: " DUCKDNS_TOKEN
echo ""  # salto de línea tras input silencioso

if [ -z "$DUCKDNS_SUBDOMAIN" ] || [ -z "$DUCKDNS_TOKEN" ]; then
    echo "Error: Subdominio y token son obligatorios"
    exit 1
fi

DUCKDNS_DOMAIN="${DUCKDNS_SUBDOMAIN}.duckdns.org"
DUCKDNS_DIR="$HOME/duckdns"

# ---- Actualizar DNS ahora ----
echo ""
echo "Actualizando registro DNS..."
RESULT=$(curl -s "https://www.duckdns.org/update?domains=${DUCKDNS_SUBDOMAIN}&token=${DUCKDNS_TOKEN}&ip=")

if [ "$RESULT" = "OK" ]; then
    echo "DNS actualizado correctamente: ${DUCKDNS_DOMAIN}"
else
    echo "Error actualizando DNS. Verifica tu subdominio y token."
    echo "Respuesta recibida: ${RESULT}"
    exit 1
fi

# ---- Crear script de actualización para cron ----
mkdir -p "$DUCKDNS_DIR"

cat > "$DUCKDNS_DIR/duck.sh" << EOF
#!/bin/bash
RESULT=\$(curl -s "https://www.duckdns.org/update?domains=${DUCKDNS_SUBDOMAIN}&token=${DUCKDNS_TOKEN}&ip=")
TIMESTAMP=\$(date '+%Y-%m-%d %H:%M:%S')

if [ "\$RESULT" = "OK" ]; then
    echo "\$TIMESTAMP OK" >> ${DUCKDNS_DIR}/duck.log
else
    echo "\$TIMESTAMP ERROR: \$RESULT" >> ${DUCKDNS_DIR}/duck.log
fi

# Rotar log si supera 1MB
if [ -f "${DUCKDNS_DIR}/duck.log" ] && [ \$(wc -c < "${DUCKDNS_DIR}/duck.log") -gt 1048576 ]; then
    mv "${DUCKDNS_DIR}/duck.log" "${DUCKDNS_DIR}/duck.log.bak"
fi
EOF

chmod +x "$DUCKDNS_DIR/duck.sh"

# ---- Instalar cron job (cada 5 minutos) ----
echo "Configurando cron job..."
CRON_JOB="*/5 * * * * ${DUCKDNS_DIR}/duck.sh"

# Evitar duplicados
(crontab -l 2>/dev/null | grep -v "duckdns" ; echo "$CRON_JOB") | crontab -

echo "Cron job instalado (actualización cada 5 minutos)"

# ---- Actualizar nginx server_name ----
# Usar ruta absoluta basada en la ubicación real del script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGINX_CONF="${SCRIPT_DIR}/nginx/conf.d/backend.conf"

if [ -f "$NGINX_CONF" ]; then
    echo "Actualizando nginx server_name..."

    # Detectar si ya tiene un server_name configurado (no el catch-all _)
    CURRENT=$(grep -oP '(?<=server_name ).*(?=;)' "$NGINX_CONF" | head -1 || true)

    if [ "$CURRENT" = "$DUCKDNS_DOMAIN" ]; then
        echo "nginx ya tiene el dominio correcto, omitiendo..."
    elif [ "$CURRENT" = "_" ] || [ -z "$CURRENT" ]; then
        sed -i "s/server_name _;/server_name ${DUCKDNS_DOMAIN};/g" "$NGINX_CONF"
        echo "nginx configurado con dominio: ${DUCKDNS_DOMAIN}"
    else
        echo "ADVERTENCIA: nginx ya tiene server_name '${CURRENT}'."
        read -p "¿Reemplazar con '${DUCKDNS_DOMAIN}'? (s/N): " CONFIRM
        if [[ "$CONFIRM" =~ ^[sS]$ ]]; then
            sed -i "s/server_name ${CURRENT};/server_name ${DUCKDNS_DOMAIN};/g" "$NGINX_CONF"
            echo "nginx actualizado a: ${DUCKDNS_DOMAIN}"
        else
            echo "nginx no modificado."
        fi
    fi
else
    echo "ADVERTENCIA: No se encontró nginx/conf.d/backend.conf en ${SCRIPT_DIR}"
    echo "Actualiza server_name manualmente con: ${DUCKDNS_DOMAIN}"
fi

# ---- Generar certificado autofirmado con SAN ----
# SAN (subjectAltName) es requerido por navegadores modernos desde 2017.
# Sin él, el certificado es rechazado aunque el CN sea correcto.
echo "Regenerando certificado autofirmado con SAN..."
sudo mkdir -p /etc/nginx/ssl
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/selfsigned.key \
    -out /etc/nginx/ssl/selfsigned.crt \
    -subj "/C=CO/ST=State/L=City/O=IMA/CN=${DUCKDNS_DOMAIN}" \
    -addext "subjectAltName=DNS:${DUCKDNS_DOMAIN}"

echo "Certificado generado para: ${DUCKDNS_DOMAIN} (válido 365 días)"

# ---- Guardar configuración ----
cat > "$DUCKDNS_DIR/config" << EOF
DUCKDNS_SUBDOMAIN=${DUCKDNS_SUBDOMAIN}
DUCKDNS_TOKEN=${DUCKDNS_TOKEN}
DUCKDNS_DOMAIN=${DUCKDNS_DOMAIN}
EOF

chmod 600 "$DUCKDNS_DIR/config"

echo ""
echo "=== DuckDNS configurado ==="
echo "Dominio:      https://${DUCKDNS_DOMAIN}"
echo "Script cron:  ${DUCKDNS_DIR}/duck.sh"
echo "Log cron:     ${DUCKDNS_DIR}/duck.log"
echo "Config:       ${DUCKDNS_DIR}/config"
echo ""
echo "Si ya tienes la app desplegada, reinicia nginx:"
echo "  docker compose restart nginx"