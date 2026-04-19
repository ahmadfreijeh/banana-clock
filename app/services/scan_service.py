import io
import uuid
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.scan import DAYS_LABEL, STAGE_INDEX_MAP, Scan
from app.services.predict import predict


async def create_scan(image_bytes: bytes, user_id: uuid.UUID, db: AsyncSession) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = predict(image)

    ripeness = result["ripeness"]
    scan = Scan(
        user_id=user_id,
        scan_date=datetime.now(timezone.utc),
        ripeness=ripeness,
        stage_index=STAGE_INDEX_MAP[ripeness],
    )
    db.add(scan)
    await db.commit()
    await db.refresh(scan)

    return {
        "id": scan.id,
        "user_id": scan.user_id,
        "scan_date": scan.scan_date,
        "ripeness": scan.ripeness,
        "stage_index": scan.stage_index,
        "days_until_inedible": result["days_until_inedible"],
    }


async def get_scans(user_id: uuid.UUID, db: AsyncSession) -> list[dict]:
    result = await db.execute(
        select(Scan).where(Scan.user_id == user_id).order_by(Scan.scan_date.desc())
    )
    scans = result.scalars().all()
    return [
        {
            "id": s.id,
            "user_id": s.user_id,
            "scan_date": s.scan_date,
            "ripeness": s.ripeness,
            "stage_index": s.stage_index,
            "days_until_inedible": DAYS_LABEL[s.ripeness],
        }
        for s in scans
    ]


async def predict_inedible_day(user_id: uuid.UUID, db: AsyncSession) -> dict:
    """
    Predicts when the user's banana will become inedible using linear regression.

    How it works:
      1. Fetches all scans for the user, ordered by date (oldest first).
      2. Converts scan dates to relative day numbers (day 0 = first scan).
      3. Fits a straight line through the (day, stage_index) data points using
         numpy's polyfit — stage_index goes from 1 (unripe) to 4 (inedible).
      4. Solves for the day when the fitted line reaches stage 4 (inedible).
      5. Returns days_left (from the last scan) and the absolute predicted day number.

    Requires at least 2 scans. Raises ValueError if there is no progression
    (flat line / zero slope), which would make the prediction undefined.
    """
    result = await db.execute(
        select(Scan).where(Scan.user_id == user_id).order_by(Scan.scan_date.asc())
    )
    scans = result.scalars().all()

    if len(scans) < 2:
        raise ValueError("Need at least 2 scans to predict")

    first_date = scans[0].scan_date
    days = [(s.scan_date - first_date).days for s in scans]
    stages = [s.stage_index for s in scans]

     # Check if all days are the same
    if len(set(days)) < 2:
        raise ValueError("Cannot predict: all scans have the same date, no time progression detected")
    
    # Check if all stages are the same
    if len(set(stages)) < 2:
        raise ValueError("Cannot predict: all scans have the same ripeness stage, no progression detected")


    coeffs = np.polyfit(days, stages, 1)

    if coeffs[0] == 0:
        raise ValueError("Cannot predict: no progression detected")

    predicted_day = (4 - coeffs[1]) / coeffs[0]
    days_left = predicted_day - days[-1]

    return {
        "days_left": round(float(days_left), 1),
        "predicted_inedible_day": round(float(predicted_day), 1),
        "scans": [
            {
                "date": s.scan_date.strftime("%Y-%m-%d"),
                "ripeness": s.ripeness,
                "stage": s.stage_index,
            }
            for s in scans
        ],
    }
