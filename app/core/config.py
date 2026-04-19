import os

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    """Read from env first, then fall back to Streamlit secrets if available."""
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


DATABASE_URL: str = _get("DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/banana_clock_db")
JWT_SECRET: str = _get("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM: str = _get("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES: int = int(_get("JWT_EXPIRE_MINUTES", "60"))