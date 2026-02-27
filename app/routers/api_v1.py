"""Router principal v1 con todos los endpoints."""

import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import numpy as np

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import get_settings
from app.core.security import verify_api_key, get_current_user
from app.core.exceptions import ModelNotLoadedException, PredictionException
from app.core.logging import get_logger
from app.schemas.health import HealthResponse, ModelStatus
from app.schemas.model import ModelInfo, ModelArchitecture, TrainingInfo, DataInfo, ModelMetrics
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    ForecastRequest,
    ForecastResponse,
    ForecastStep,
    DiagnosisRequest,
    DiagnosisResponse,
    DiagnosisInfo,
    AlertLevel
)
from app.schemas.prediction import TimeSeriesPoint
from app.schemas.metrics import MetricsResponse, AllModelMetrics, OnlineMetrics, ErrorMetrics
from app.schemas.alerts import AlertRequest, AlertResponse, WeatherAlertInfo
from app.services.model_service import get_model_service
from app.services.prediction_service import PredictionService
from app.services.diagnosis_service import DiagnosisService
from app.services.smart_diagnosis_service import SmartDiagnosisService, get_smart_diagnosis_service
from app.services.metrics_service import get_metrics_service
from app.services.xgboost_service import get_xgboost_service
from app.services.feature_utils import prepare_features as _shared_prepare_features
from app.services.alert_service import get_alert_service
from app.services.subscription_service import get_subscription_service
from app.services.alert_history_service import get_alert_history_service
from app.schemas.subscription import (
    SubscriptionCreate,
    SubscriptionUpdate,
    SubscriptionResponse,
    SubscriptionListResponse,
    SubscriptionDeleteResponse
)
from app.schemas.alert_history import (
    AlertHistoryRequest,
    AlertHistoryResponse,
    AlertHistoryEntry as AlertHistoryEntrySchema,
    AlertStatisticsResponse
)

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()

# Instancias de servicios
model_service = get_model_service()
prediction_service = PredictionService()
diagnosis_service = DiagnosisService()
smart_diagnosis_service = get_smart_diagnosis_service()
metrics_service = get_metrics_service()
xgboost_service = get_xgboost_service()
alert_service = get_alert_service()
subscription_service = get_subscription_service()
alert_history_service = get_alert_history_service()




def prepare_xgboost_features(lookback_data: List) -> np.ndarray:
    """Delega a feature_utils.prepare_features (módulo compartido)."""
    return _shared_prepare_features(lookback_data)


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Verifica el estado del servicio y del modelo.
    """
    model_status = ModelStatus.LOADED if model_service.is_loaded() else ModelStatus.NOT_LOADED
    
    return HealthResponse(
        status="healthy" if model_service.is_loaded() else "degraded",
        timestamp=datetime.now(timezone.utc),
        version=settings.api_version,
        model_status=model_status,
        model_loaded_at=model_service.get_loaded_at(),
        uptime_seconds=metrics_service.get_uptime()
    )


@router.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info(_: dict = Depends(get_current_user)):
    """
    Obtiene información de ambos modelos (LSTM y XGBoost).
    
    Incluye hiperparámetros y métricas de evaluación de ambos modelos.
    """
    try:
        # Obtener información de ambos modelos desde model_service
        lstm_info = model_service.get_lstm_info()
        xgboost_info = model_service.get_xgboost_info()
        
        # Usar timestamp del LSTM como timestamp principal
        timestamp = lstm_info.get("timestamp", xgboost_info.get("timestamp", "unknown"))
        
        return ModelInfo(
            timestamp=timestamp,
            lstm=lstm_info,
            xgboost=xgboost_info
        )
        
    except ModelNotLoadedException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: PredictionRequest,
    _: dict = Depends(get_current_user)
):
    """
    Predicción t+1.
    
    Recibe ventana de observación y retorna predicción de precipitación,
    probabilidad de evento, diagnóstico y recomendaciones.
    """
    start_time = time.time()
    
    try:
        # Validar que hay datos
        if not request.lookback_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="lookback_data no puede estar vacío"
            )
        
        # Predicción
        precip_pred, rain_prob = prediction_service.predict(request.lookback_data)
        
        # Diagnóstico ADAPTATIVO (usa correlaciones reales)
        current = request.lookback_data[-1]
        previous = request.lookback_data[:-1] if len(request.lookback_data) > 1 else []
        
        alert_level, triggered_rules, recommendation, scores = smart_diagnosis_service.diagnose_adaptive(
            current=current,
            previous=previous if previous else None,
            predicted_precip=precip_pred
        )
        
        # Latencia
        latency_ms = (time.time() - start_time) * 1000
        
        # Registrar métrica
        metrics_service.record_prediction(latency_ms)
        
        return PredictionResponse(
            prediction_mm_hr=precip_pred,
            rain_event_prob=rain_prob,
            diagnosis=DiagnosisInfo(
                level=alert_level,
                triggered_rules=triggered_rules,
                recommendation=recommendation
            ),
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc)
        )
        
    except PredictionException as e:
        metrics_service.record_error("prediction_error")
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        metrics_service.record_error("unknown_error")
        logger.exception("Error inesperado en predicción")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}"
        )


@router.post("/forecast", response_model=ForecastResponse, tags=["Prediction"])
async def forecast(
    request: ForecastRequest,
    _: dict = Depends(get_current_user)
):
    """
    Pronóstico multi-paso.
    
    Genera predicciones para múltiples pasos futuros.
    """
    start_time = time.time()
    
    try:
        # Validar que hay datos
        if not request.lookback_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="lookback_data no puede estar vacío"
            )
        
        horizon = min(request.horizon, settings.max_forecast_horizon)
        forecast_steps = []
        
        # Usar predicción multi-paso nativa del modelo LSTM
        multistep_results = prediction_service.predict_multistep(request.lookback_data)
        
        # Diagnóstico base
        current = request.lookback_data[-1]
        previous = request.lookback_data[:-1] if len(request.lookback_data) > 1 else []
        last_ts = request.lookback_data[-1].timestamp
        
        # Tomar hasta horizon pasos del resultado nativo
        n_native_steps = len(multistep_results)
        
        for step in range(min(horizon, n_native_steps)):
            precip_pred, rain_prob = multistep_results[step]
            
            alert_level, _, _, _ = smart_diagnosis_service.diagnose_adaptive(
                current=current,
                previous=previous if previous else None,
                predicted_precip=precip_pred
            )
            
            next_ts = last_ts + timedelta(hours=step + 1)
            
            forecast_steps.append(ForecastStep(
                timestamp=next_ts,
                prediction_mm_hr=precip_pred,
                rain_event_prob=rain_prob,
                diagnosis_level=alert_level
            ))
        
        # Si el horizon solicitado excede los pasos nativos, usar autoregresivo
        if horizon > n_native_steps:
            current_lookback = request.lookback_data.copy()
            for step in range(n_native_steps, horizon):
                precip_pred, rain_prob = prediction_service.predict(current_lookback)
                
                alert_level, _, _, _ = smart_diagnosis_service.diagnose_adaptive(
                    current=current,
                    previous=previous if previous else None,
                    predicted_precip=precip_pred
                )
                
                next_ts = last_ts + timedelta(hours=step + 1)
                
                forecast_steps.append(ForecastStep(
                    timestamp=next_ts,
                    prediction_mm_hr=precip_pred,
                    rain_event_prob=rain_prob,
                    diagnosis_level=alert_level
                ))
                
                next_point = TimeSeriesPoint(
                    timestamp=next_ts,
                    rh_2m_pct=current.rh_2m_pct,
                    temp_2m_c=current.temp_2m_c,
                    wind_speed_2m_ms=current.wind_speed_2m_ms,
                    wind_dir_2m_deg=current.wind_dir_2m_deg,
                )
                current_lookback = current_lookback[1:] + [next_point]
            
        latency_ms = (time.time() - start_time) * 1000
        metrics_service.record_forecast(latency_ms)
        
        return ForecastResponse(
            forecast=forecast_steps,
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        metrics_service.record_error("forecast_error")
        logger.exception("Error en pronóstico")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en pronóstico: {str(e)}"
        )


@router.post("/diagnosis", response_model=DiagnosisResponse, tags=["Diagnosis"])
async def diagnosis(
    request: DiagnosisRequest,
    _: dict = Depends(get_current_user)
):
    """
    Diagnóstico basado en reglas.
    
    Evalúa condiciones meteorológicas y retorna nivel de alerta
    con recomendaciones operacionales.
    """
    try:
        alert_level, triggered_rules, recommendation = diagnosis_service.diagnose(
            current=request.current_conditions,
            previous=request.previous_conditions,
            predicted_precip=request.predicted_precip_mm_hr
        )
        
        metrics_service.record_diagnosis()
        
        return DiagnosisResponse(
            level=alert_level,
            triggered_rules=triggered_rules,
            recommendation=recommendation,
            metrics={
                "rh_2m_pct": request.current_conditions.rh_2m_pct,
                "temp_2m_c": request.current_conditions.temp_2m_c,
                "wind_speed_2m_ms": request.current_conditions.wind_speed_2m_ms
            },
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        metrics_service.record_error("diagnosis_error")
        logger.exception("Error en diagnóstico")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics(_: dict = Depends(get_current_user)):
    """
    Obtiene métricas del servicio.
    
    Incluye métricas offline del modelo y métricas online
    de requests y latencias.
    """
    try:
        metrics_data = metrics_service.get_metrics()
        
        return MetricsResponse(
            timestamp=metrics_data["timestamp"],
            model_metrics=AllModelMetrics(**metrics_data["model_metrics"]),
            online_metrics=OnlineMetrics(**metrics_data["online_metrics"]),
            error_metrics=ErrorMetrics(**metrics_data["error_metrics"]),
            uptime_seconds=metrics_data["uptime_seconds"]
        )
        
    except Exception as e:
        logger.exception("Error obteniendo métricas")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/diagnosis/adaptive", response_model=DiagnosisResponse, tags=["Diagnosis"])
async def diagnosis_adaptive(
    request: DiagnosisRequest,
    _: dict = Depends(get_current_user)
):
    """
    Diagnóstico ADAPTATIVO con umbrales dinámicos.
    
    A diferencia de /diagnosis estático, este endpoint:
    - Calcula scores de riesgo por cada factor
    - Usa correlaciones reales (precip vs RH, precip vs temp)
    - Genera umbrales dinámicos según condiciones
    - Proporciona nivel de confianza
    """
    try:
        alert_level, triggered_rules, recommendation, scores = smart_diagnosis_service.diagnose_adaptive(
            current=request.current_conditions,
            previous=request.previous_conditions,
            predicted_precip=request.predicted_precip_mm_hr
        )
        
        metrics_service.record_diagnosis()
        
        # Añadir scores al response
        metrics = {
            "rh_2m_pct": request.current_conditions.rh_2m_pct,
            "temp_2m_c": request.current_conditions.temp_2m_c,
            "wind_speed_2m_ms": request.current_conditions.wind_speed_2m_ms,
            "risk_scores": scores  # Scores de cada factor
        }
        
        return DiagnosisResponse(
            level=alert_level,
            triggered_rules=triggered_rules,
            recommendation=recommendation,
            metrics=metrics,
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        metrics_service.record_error("adaptive_diagnosis_error")
        logger.exception("Error en diagnóstico adaptativo")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/model/reload", tags=["Model"])
async def reload_model(api_key: str = Depends(verify_api_key)):
    """
    Recarga el modelo sin reiniciar el servicio.
    
    Requiere API Key válida.
    """
    try:
        logger.info("Recargando modelo por request de API...")
        model_service.reload_model()
        
        return {
            "status": "success",
            "message": "Modelo recargado exitosamente",
            "loaded_at": model_service.get_loaded_at().isoformat()
        }
        
    except Exception as e:
        logger.exception("Error recargando modelo")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recargando modelo: {str(e)}"
        )


@router.post("/predict/xgboost", response_model=PredictionResponse, tags=["Prediction"])
async def predict_xgboost(
    request: PredictionRequest,
    _: dict = Depends(get_current_user)
):
    """
    Predicción t+1 con XGBoost.
    
    Alternativa más rápida al LSTM para predicción de precipitación.
    """
    start_time = time.time()
    
    try:
        # Cargar modelo si no está cargado
        if not xgboost_service.is_loaded():
            xgboost_service.load_model()
        
        # Preparar features usando el mismo feature engineering que LSTM
        features = prepare_xgboost_features(request.lookback_data)
        
        # XGBoost espera solo el último punto (no secuencia como LSTM)
        if features.shape[0] > 0:
            features = features[-1]  # Tomar solo el último punto
        else:
            raise ValueError("No hay datos válidos para predicción")
        
        # Predicción
        precip_pred, rain_prob = xgboost_service.predict(features)
        
        # Diagnóstico
        current = request.lookback_data[-1]
        previous = request.lookback_data[:-1] if len(request.lookback_data) > 1 else []
        
        alert_level, triggered_rules, recommendation, scores = smart_diagnosis_service.diagnose_adaptive(
            current=current,
            previous=previous if previous else None,
            predicted_precip=precip_pred
        )
        
        # Latencia
        latency_ms = (time.time() - start_time) * 1000
        
        metrics_service.record_prediction(latency_ms)
        
        return PredictionResponse(
            prediction_mm_hr=precip_pred,
            rain_event_prob=rain_prob,
            diagnosis=DiagnosisInfo(
                level=alert_level,
                triggered_rules=triggered_rules,
                recommendation=recommendation
            ),
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc)
        )
        
    except ModelNotLoadedException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        metrics_service.record_error("xgboost_prediction_error")
        logger.exception("Error en predicción XGBoost")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/predict/hybrid", response_model=PredictionResponse, tags=["Prediction"])
async def predict_hybrid(
    request: PredictionRequest,
    _: dict = Depends(get_current_user)
):
    """
    Predicción híbrida LSTM + XGBoost.

    Combina las predicciones de ambos modelos con promedio pesado
    (pesos configurables en configs/hyperparameters.json).
    """
    start_time = time.time()

    try:
        if not request.lookback_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="lookback_data no puede estar vacío"
            )

        # Predicción LSTM (t+1)
        precip_lstm, _ = prediction_service.predict(request.lookback_data)

        # Predicción XGBoost
        if not xgboost_service.is_loaded():
            xgboost_service.load_model()

        features = prepare_xgboost_features(request.lookback_data)
        if features.shape[0] > 0:
            features = features[-1]
        else:
            raise ValueError("No hay datos válidos para predicción")

        precip_xgb, _ = xgboost_service.predict(features)

        # Combinar con pesos (defaults 0.5/0.5, leídos de config)
        from src.utils.hyperparams import get_ensemble_defaults
        ens = get_ensemble_defaults()
        w_lstm = ens.get("w_lstm", 0.5)
        w_xgb = ens.get("w_xgboost", 0.5)

        precip_pred = w_lstm * precip_lstm + w_xgb * precip_xgb
        precip_pred = max(0.0, precip_pred)

        # Probabilidad de lluvia sobre la predicción combinada
        from app.services.feature_utils import calculate_rain_probability
        rain_prob = calculate_rain_probability(precip_pred)

        # Diagnóstico
        current = request.lookback_data[-1]
        previous = request.lookback_data[:-1] if len(request.lookback_data) > 1 else []

        alert_level, triggered_rules, recommendation, scores = smart_diagnosis_service.diagnose_adaptive(
            current=current,
            previous=previous if previous else None,
            predicted_precip=precip_pred
        )

        latency_ms = (time.time() - start_time) * 1000
        metrics_service.record_prediction(latency_ms)

        return PredictionResponse(
            prediction_mm_hr=precip_pred,
            rain_event_prob=rain_prob,
            diagnosis=DiagnosisInfo(
                level=alert_level,
                triggered_rules=triggered_rules,
                recommendation=recommendation
            ),
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc)
        )

    except ModelNotLoadedException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        metrics_service.record_error("hybrid_prediction_error")
        logger.exception("Error en predicción híbrida")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/alerts/evaluate", response_model=AlertResponse, tags=["Alerts"])
async def evaluate_alerts(
    request: AlertRequest,
    _: dict = Depends(get_current_user)
):
    """
    Evalúa condiciones climáticas y genera alertas.
    
    Analiza las variables climáticas actuales y previas para detectar
    condiciones que requieren atención (niveles MEDIA, ALTA, CRITICA).
    """
    try:
        # Evaluar condiciones y generar alertas
        alerts = alert_service.evaluate_conditions(
            current=request.current_conditions,
            previous=request.previous_conditions
        )
        
        # Convertir alertas a formato de respuesta
        alert_infos = [
            WeatherAlertInfo(**alert.to_dict())
            for alert in alerts
        ]
        
        # Verificar si hay alertas de nivel MEDIO o superior
        has_media_or_higher = any(
            AlertLevel(alert.level).severity >= AlertLevel.MEDIO.severity
            for alert in alerts
        )
        
        # Verificar si hay alertas de nivel ALTO o superior
        has_alta_or_higher = any(
            AlertLevel(alert.level).severity >= AlertLevel.ALTO.severity
            for alert in alerts
        )
        
        return AlertResponse(
            alerts=alert_infos,
            total_alerts=len(alert_infos),
            has_media_or_higher=has_media_or_higher,
            has_alta_or_higher=has_alta_or_higher,
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        metrics_service.record_error("alert_evaluation_error")
        logger.exception("Error evaluando alertas")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/alerts/active", response_model=AlertResponse, tags=["Alerts"])
async def get_active_alerts(_: dict = Depends(get_current_user)):
    """
    Obtiene las alertas activas actuales.
    
    Retorna todas las alertas que fueron generadas en la última evaluación.
    """
    try:
        active_alerts = alert_service.get_active_alerts()
        
        # Convertir a WeatherAlertInfo
        alert_infos = [
            WeatherAlertInfo(**alert)
            for alert in active_alerts
        ]
        
        # Verificar niveles
        has_media_or_higher = any(
            AlertLevel(alert["level"]).severity >= AlertLevel.MEDIO.severity
            for alert in active_alerts
        )
        
        has_alta_or_higher = any(
            AlertLevel(alert["level"]).severity >= AlertLevel.ALTO.severity
            for alert in active_alerts
        )
        
        return AlertResponse(
            alerts=alert_infos,
            total_alerts=len(alert_infos),
            has_media_or_higher=has_media_or_higher,
            has_alta_or_higher=has_alta_or_higher,
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.exception("Error obteniendo alertas activas")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# ENDPOINTS DE SUSCRIPCIONES (RF9)
# ============================================================================

@router.post("/subscriptions", response_model=SubscriptionResponse, tags=["Subscriptions"])
async def create_subscription(
    request: SubscriptionCreate,
    _: dict = Depends(get_current_user)
):
    """
    Crea una nueva suscripción a alertas.
    
    Permite configurar:
    - Nivel mínimo de alertas (baja, media, alta, critica)
    - Variables específicas a monitorear (opcional, [] = todas)
    """
    try:
        subscription = subscription_service.create_subscription(
            user_id=request.user_id,
            min_level=request.min_level,
            variables=request.variables,
            notification_frequency=request.notification_frequency
        )
        
        return SubscriptionResponse(**subscription.to_dict())
        
    except Exception as e:
        logger.exception("Error creando suscripción")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/subscriptions/{user_id}", response_model=SubscriptionListResponse, tags=["Subscriptions"])
async def get_user_subscriptions(
    user_id: str,
    enabled_only: bool = False,
    _: dict = Depends(get_current_user)
):
    """
    Obtiene todas las suscripciones de un usuario.
    
    Args:
        user_id: ID del usuario
        enabled_only: Si True, solo retorna suscripciones activas
    """
    try:
        subscriptions = subscription_service.get_user_subscriptions(
            user_id=user_id,
            enabled_only=enabled_only
        )
        
        subscription_dicts = [sub.to_dict() for sub in subscriptions]
        
        return SubscriptionListResponse(
            subscriptions=[SubscriptionResponse(**sub) for sub in subscription_dicts],
            total=len(subscription_dicts)
        )
        
    except Exception as e:
        logger.exception("Error obteniendo suscripciones")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.patch("/subscriptions/{subscription_id}", response_model=SubscriptionResponse, tags=["Subscriptions"])
async def update_subscription(
    subscription_id: str,
    request: SubscriptionUpdate,
    _: dict = Depends(get_current_user)
):
    """
    Actualiza una suscripción existente.
    
    Permite modificar:
    - Nivel mínimo de alertas
    - Variables monitoreadas
    - Estado (enabled/disabled)
    """
    try:
        subscription = subscription_service.update_subscription(
            subscription_id=subscription_id,
            min_level=request.min_level,
            variables=request.variables,
            enabled=request.enabled,
            notification_frequency=request.notification_frequency
        )
        
        if not subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suscripción {subscription_id} no encontrada"
            )
        
        return SubscriptionResponse(**subscription.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error actualizando suscripción")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/subscriptions/{subscription_id}", response_model=SubscriptionDeleteResponse, tags=["Subscriptions"])
async def delete_subscription(
    subscription_id: str,
    _: dict = Depends(get_current_user)
):
    """
    Elimina una suscripción.
    
    La suscripción se elimina permanentemente.
    """
    try:
        success = subscription_service.delete_subscription(subscription_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suscripción {subscription_id} no encontrada"
            )
        
        return SubscriptionDeleteResponse(
            success=True,
            message="Suscripción eliminada exitosamente",
            subscription_id=subscription_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error eliminando suscripción")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/subscriptions/{subscription_id}/disable", response_model=SubscriptionDeleteResponse, tags=["Subscriptions"])
async def disable_subscription(
    subscription_id: str,
    _: dict = Depends(get_current_user)
):
    """
    Desactiva una suscripción sin eliminarla.
    
    Puede reactivarse posteriormente usando PATCH.
    """
    try:
        success = subscription_service.disable_subscription(subscription_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suscripción {subscription_id} no encontrada"
            )
        
        return SubscriptionDeleteResponse(
            success=True,
            message="Suscripción desactivada exitosamente",
            subscription_id=subscription_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error desactivando suscripción")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# ENDPOINTS DE HISTORIAL (RF10)
# ============================================================================

@router.get("/alerts/history", response_model=AlertHistoryResponse, tags=["Alert History"])
async def get_alert_history(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    level: Optional[str] = None,
    variable: Optional[str] = None,
    limit: int = 100,
    _: dict = Depends(get_current_user)
):
    """
    Obtiene el historial de alertas con filtros opcionales.
    
    Args:
        from_date: Fecha inicial en formato ISO (ej: 2025-01-01T00:00:00)
        to_date: Fecha final en formato ISO
        level: Filtrar por nivel (baja, media, alta, critica)
        variable: Filtrar por variable
        limit: Número máximo de resultados (default: 100)
    """
    try:
        # Parsear fechas si se proporcionan
        from_dt = datetime.fromisoformat(from_date) if from_date else None
        to_dt = datetime.fromisoformat(to_date) if to_date else None
        
        # Obtener historial
        entries = alert_history_service.get_history(
            from_date=from_dt,
            to_date=to_dt,
            level=level,
            variable=variable,
            limit=limit
        )
        
        # Convertir a schema
        alert_entries = [
            AlertHistoryEntrySchema(**entry.to_dict())
            for entry in entries
        ]
        
        return AlertHistoryResponse(
            alerts=alert_entries,
            total=len(alert_entries),
            filters_applied={
                "from_date": from_date,
                "to_date": to_date,
                "level": level,
                "variable": variable,
                "limit": str(limit)
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato de fecha inválido: {str(e)}"
        )
    except Exception as e:
        logger.exception("Error obteniendo historial de alertas")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/alerts/history/recent", response_model=AlertHistoryResponse, tags=["Alert History"])
async def get_recent_alerts(
    hours: int = 24,
    _: dict = Depends(get_current_user)
):
    """
    Obtiene alertas recientes.
    
    Args:
        hours: Número de horas hacia atrás (default: 24)
    """
    try:
        entries = alert_history_service.get_recent_alerts(hours=hours)
        
        alert_entries = [
            AlertHistoryEntrySchema(**entry.to_dict())
            for entry in entries
        ]
        
        return AlertHistoryResponse(
            alerts=alert_entries,
            total=len(alert_entries),
            filters_applied={
                "hours": str(hours),
                "from_date": (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            }
        )
        
    except Exception as e:
        logger.exception("Error obteniendo alertas recientes")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/alerts/history/statistics", response_model=AlertStatisticsResponse, tags=["Alert History"])
async def get_alert_statistics(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    _: dict = Depends(get_current_user)
):
    """
    Obtiene estadísticas del historial de alertas.
    
    Args:
        from_date: Fecha inicial (opcional)
        to_date: Fecha final (opcional)
    """
    try:
        from_dt = datetime.fromisoformat(from_date) if from_date else None
        to_dt = datetime.fromisoformat(to_date) if to_date else None
        
        stats = alert_history_service.get_statistics(
            from_date=from_dt,
            to_date=to_dt
        )
        
        return AlertStatisticsResponse(**stats)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato de fecha inválido: {str(e)}"
        )
    except Exception as e:
        logger.exception("Error obteniendo estadísticas de alertas")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

