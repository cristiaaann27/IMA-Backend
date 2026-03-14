#!/bin/bash
set -e

echo "=== Configuración DuckDNS ==="
echo ""

# ---- Solicitar datos ----
read -p "Subdominio DuckDNS (ej: ima-api): " DUCKDNS_SUBDOMAIN
read -p "Token DuckDNS: " DUCKDNS_TOKEN

if [ -z "$DUCKDNS_SUBDOMAIN" ] || [ -z "$DUCKDNS_TOKEN" ]; then
    echo "Error: Subdominio y token son obligatorios"
    exit 1
fi

DUCKDNS_DOMAIN="${DUCKDNS_SUBDOMAIN}.duckdns.org"

# ---- Actualizar DNS ahora ----
echo ""
echo "Actualizando registro DNS..."
RESULT=$(curl -s "https://www.duckdns.org/update?domains=${DUCKDNS_SUBDOMAIN}&token=${DUCKDNS_TOKEN}&ip=")

if [ "$RESULT" = "OK" ]; then
    echo "DNS actualizado correctamente: ${DUCKDNS_DOMAIN}"
else
    echo "Error actualizando DNS. Verifica tu subdominio y token."
    exit 1
fi

# ---- Crear script de actualización para cron ----
DUCKDNS_DIR="$HOME/duckdns"
mkdir -p "$DUCKDNS_DIR"

cat > "$DUCKDNS_DIR/duck.sh" << EOF
#!/bin/bash
echo url="https://www.duckdns.org/update?domains=${DUCKDNS_SUBDOMAIN}&token=${DUCKDNS_TOKEN}&ip=" | curl -k -o ${DUCKDNS_DIR}/duck.log -K -
EOF

chmod +x "$DUCKDNS_DIR/duck.sh"

# ---- Instalar cron job (cada 5 minutos) ----
echo "Configurando cron job..."
CRON_JOB="*/5 * * * * ${DUCKDNS_DIR}/duck.sh >/dev/null 2>&1"

# Evitar duplicados
(crontab -l 2>/dev/null | grep -v "duckdns" ; echo "$CRON_JOB") | crontab -

echo "Cron job instalado (actualización cada 5 minutos)"

# ---- Actualizar nginx server_name ----
NGINX_CONF="$(dirname "$0")/nginx/conf.d/backend.conf"
if [ -f "$NGINX_CONF" ]; then
    echo "Actualizando nginx server_name..."
    sed -i "s/server_name _;/server_name ${DUCKDNS_DOMAIN};/g" "$NGINX_CONF"
    echo "nginx configurado con dominio: ${DUCKDNS_DOMAIN}"
fi

# ---- Actualizar certificado autofirmado con el dominio ----
echo "Regenerando certificado autofirmado con dominio..."
sudo mkdir -p /etc/nginx/ssl
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/selfsigned.key \
    -out /etc/nginx/ssl/selfsigned.crt \
    -subj "/C=CO/ST=State/L=City/O=IMA/CN=${DUCKDNS_DOMAIN}"

# ---- Guardar configuración ----
cat > "$DUCKDNS_DIR/config" << EOF
DUCKDNS_SUBDOMAIN=${DUCKDNS_SUBDOMAIN}
DUCKDNS_TOKEN=${DUCKDNS_TOKEN}
DUCKDNS_DOMAIN=${DUCKDNS_DOMAIN}
EOF

chmod 600 "$DUCKDNS_DIR/config"

echo ""
echo "=== DuckDNS configurado ==="
echo "Dominio: https://${DUCKDNS_DOMAIN}"
echo "Script cron: ${DUCKDNS_DIR}/duck.sh"
echo "Config: ${DUCKDNS_DIR}/config"
echo ""
echo "Si ya tienes la app desplegada, reinicia nginx:"
echo "  docker compose restart nginx"
