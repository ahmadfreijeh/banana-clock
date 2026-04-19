from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import DATABASE_URL

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,   # test connection health before each checkout
    pool_recycle=1800,    # recycle connections older than 30 min
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
