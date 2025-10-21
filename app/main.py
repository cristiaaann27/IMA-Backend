import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.exceptions import (
    APIException,
    api_exception_handler,
    http_exception_handler,
    general_exception_handler
)
from app.core.logging import get_logger, request_id_context
from app.routers.api_v1 import router as api_v1_router
from app.services.model_service import get_model_service
from app.services.xgboost_service import get_xgboost_service

from fastapi import HTTPException

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Iniciando aplicación...")
    logger.info(f"Configuración: {settings.api_title} v{settings.api_version}")
    
    # Cargar modelo LSTM
    try:
        model_service = get_model_service()
        model_service.load_model()
        logger.info("[OK] Modelo LSTM cargado exitosamente")
    except Exception as e:
        logger.error(f"[ERROR] Error cargando modelo LSTM: {e}")
        logger.warning("La aplicación iniciará pero el modelo LSTM no está disponible")
    
    # Cargar modelo XGBoost
    try:
        xgboost_service = get_xgboost_service()
        xgboost_service.load_model()
        logger.info("[OK] Modelo XGBoost cargado exitosamente")
    except Exception as e:
        logger.error(f"[ERROR] Error cargando modelo XGBoost: {e}")
        logger.warning("La aplicación iniciará pero el modelo XGBoost no está disponible")
    
    yield
    
    # Shutdown
    logger.info("Cerrando aplicación...")


# Crear aplicación
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    openapi_url=f"{settings.api_prefix}/openapi.json"
)


# Configuración de CORS
# En desarrollo permite todos los orígenes, en producción usa variable de entorno
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins == "*":
    origins = ["*"]
else:
    origins = [origin.strip() for origin in allowed_origins.split(",")]

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False if origins == ["*"] else True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# Middleware para request-id
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Añade request-id a cada request."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request_id_context.set(request_id)
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# Middleware para logging de requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logea cada request."""
    logger.info(
        f"{request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None
        }
    )
    
    response = await call_next(request)
    
    logger.info(
        f"Response: {response.status_code}",
        extra={"status_code": response.status_code}
    )
    
    return response


# Exception handlers
app.add_exception_handler(APIException, api_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


# Incluir routers
app.include_router(api_v1_router, prefix=settings.api_prefix)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint con información básica."""
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "docs": f"{settings.api_prefix}/docs",
        "health": f"{settings.api_prefix}/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )

