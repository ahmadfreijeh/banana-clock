from fastapi import APIRouter

from app.api.routes import health, predict

router = APIRouter()

router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(predict.router, prefix="/predict", tags=["predict"])
