import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from app.services.predict import predict

router = APIRouter()


@router.post("")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = predict(image)
    return result
