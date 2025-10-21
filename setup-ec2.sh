#!/bin/bash
set -e

echo "=== Configuración inicial EC2 ==="

# Actualizar sistema
echo "Actualizando sistema..."
sudo apt-get update
sudo apt-get upgrade -y

# Instalar Docker
echo "Instalando Docker..."
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Instalar Docker Compose
echo "Instalando Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Agregar usuario al grupo docker
echo "Configurando permisos Docker..."
sudo usermod -aG docker $USER

# Instalar Git
echo "Instalando Git..."
sudo apt-get install -y git

# Crear directorios necesarios
echo "Creando estructura de directorios..."
mkdir -p ~/proyecto/models
mkdir -p ~/proyecto/data/{raw,processed,curated,predictions}
mkdir -p ~/proyecto/reports

echo "=== Configuración completada ==="
echo "IMPORTANTE: Cierra sesión y vuelve a entrar para aplicar permisos de Docker"
echo "Luego clona el repositorio y ejecuta ./deploy.sh"
