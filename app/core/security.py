from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from .config import get_settings
from .logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:

    if api_key is None:
        logger.warning("API Key no proporcionada")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key requerida",
            headers={settings.api_key_header: "Required"}
        )
    
    if api_key != settings.api_key:
        logger.warning(f"API Key inválida: {api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida"
        )
    
    return api_key


async def get_current_user(api_key: str = Security(api_key_header)) -> dict:

    await verify_api_key(api_key)

    return {
        "user_id": "api_user",
        "roles": ["user"],
        "authenticated": True
    }