#!/bin/bash
set -e

echo "=== Entrenamiento de Modelos ==="

# Verificar que el contenedor esté corriendo
if ! docker-compose ps | grep -q "precipitation-api.*Up"; then
    echo "Error: El contenedor backend no está corriendo"
    echo "Ejecuta primero: ./deploy.sh"
    exit 1
fi

# Menú de opciones
echo ""
echo "Selecciona una opción:"
echo "1) Pipeline completo (ETL + Features + Train LSTM + XGBoost)"
echo "2) Solo ETL"
echo "3) Solo Features"
echo "4) Solo entrenar LSTM"
echo "5) Solo entrenar XGBoost"
echo "6) Evaluar LSTM"
echo "7) Comparar modelos"
echo "8) Generar predicciones"
echo "9) Acceder al contenedor (modo interactivo)"
echo ""
read -p "Opción: " option

case $option in
    1)
        echo "Ejecutando pipeline completo..."
        docker-compose exec backend python -m src all
        ;;
    2)
        echo "Ejecutando ETL..."
        docker-compose exec backend python -m src etl
        ;;
    3)
        echo "Ejecutando construcción de features..."
        docker-compose exec backend python -m src features
        ;;
    4)
        echo "Entrenando LSTM..."
        docker-compose exec backend python -m src train-lstm
        ;;
    5)
        echo "Entrenando XGBoost..."
        docker-compose exec backend python -m src train-xgboost
        ;;
    6)
        echo "Evaluando LSTM..."
        docker-compose exec backend python -m src eval-lstm
        ;;
    7)
        echo "Comparando modelos..."
        docker-compose exec backend python -m src compare
        ;;
    8)
        echo "Generando predicciones..."
        docker-compose exec backend python -m src predict
        ;;
    9)
        echo "Accediendo al contenedor..."
        docker-compose exec backend bash
        ;;
    *)
        echo "Opción inválida"
        exit 1
        ;;
esac

echo ""
echo "=== Completado ==="
