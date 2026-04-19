from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    import os
    from app.services.train import train_model
    if os.getenv("RETRAIN", "true").strip().lower() == "true":
        print("Starting training...")
        await asyncio.to_thread(train_model)
    else:
        print("RETRAIN=false — skipping training, using existing model.")
    yield


app = FastAPI(title="Banana Clock", version="0.1.0", lifespan=lifespan)

app.include_router(router, prefix="/api")
