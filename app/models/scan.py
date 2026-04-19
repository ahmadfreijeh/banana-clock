import uuid
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, ForeignKey, DateTime
from app.models.base import Base, UUIDMixin, TimestampMixin


RIPENESS_VALUES = ['overripe', 'ripe', 'rotten', 'unripe']
STAGE_INDEX_MAP = {label: i + 1 for i, label in enumerate(RIPENESS_VALUES)}
DAYS_LABEL = {
    "unripe": "5-7 days until ripe, 12-14 days until inedible",
    "ripe": "Perfect now! 4-6 days until overripe",
    "overripe": "1-2 days left, eat soon!",
    "rotten": "Too late! Time to throw it away",
}


class Scan(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "scans"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    scan_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ripeness: Mapped[str] = mapped_column(String(20), nullable=False)
    stage_index: Mapped[int] = mapped_column(Integer, nullable=False)
