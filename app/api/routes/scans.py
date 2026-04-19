import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User
from app.services import scan_service

router = APIRouter()


class ScanResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    scan_date: datetime
    ripeness: str
    stage_index: int
    days_until_inedible: str


class InediblePrediction(BaseModel):
    days_left: float
    predicted_inedible_day: float
    scans: list[dict]


@router.post("", response_model=ScanResponse, status_code=status.HTTP_201_CREATED)
async def create_scan(
    file: UploadFile = File(...),
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be an image")

    image_bytes = await file.read()
    return await scan_service.create_scan(image_bytes, current_user.id, db)


@router.get("", response_model=list[ScanResponse])
async def get_user_scans(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    return await scan_service.get_scans(current_user.id, db)


@router.get("/predict-inedible-day", response_model=InediblePrediction)
async def predict_inedible_day(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        return await scan_service.predict_inedible_day(current_user.id, db)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
