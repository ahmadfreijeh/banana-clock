from fastapi import APIRouter

from app.api.routes import auth, health, scans

router = APIRouter()

router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(scans.router, prefix="/scans", tags=["scans"])
