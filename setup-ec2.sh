#!/bin/bash
set -e

echo "=== Configuración inicial EC2 (Free Tier) ==="

# ---- Swap (crítico para t2.micro/t3.micro con 1GB RAM) ----
if [ ! -f /swapfile ]; then
    echo "Creando swap de 2GB..."
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    # Optimizar swappiness para servidor
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
    sudo sysctl -p
    echo "Swap de 2GB configurado"
else
    echo "Swap ya existe, omitiendo..."
fi

# ---- Actualizar sistema ----
echo "Actualizando sistema..."
sudo apt-get update
sudo apt-get upgrade -y

# ---- Instalar Docker ----
echo "Instalando Docker..."
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# ---- Instalar Docker Compose v2 (plugin) ----
echo "Instalando Docker Compose v2..."
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# ---- Permisos Docker ----
echo "Configurando permisos Docker..."
sudo usermod -aG docker $USER

# ---- Instalar Git ----
echo "Instalando Git..."
sudo apt-get install -y git

# ---- Firewall (UFW) ----
echo "Configurando firewall..."
sudo apt-get install -y ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw --force enable
echo "Firewall configurado (puertos 22, 80, 443)"

# ---- Crear directorios (se montarán como volúmenes en Docker) ----
echo "Creando estructura de directorios..."
mkdir -p ~/models
mkdir -p ~/data/{raw,processed,curated,predictions}
mkdir -p ~/reports

# ---- Limpiar caché ----
echo "Limpiando caché..."
sudo apt-get autoremove -y
sudo apt-get clean

echo ""
echo "=== Configuración completada ==="
echo "IMPORTANTE: Cierra sesión y vuelve a entrar para aplicar permisos de Docker"
echo "Luego clona el repositorio y ejecuta ./deploy.sh"
