"""FastAPI app: loads models once at startup, mounts routers."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.models_loader import load_models, models
from app.routers import classify, duplicate, hotspots, priority, verify
from app.services import classify as classify_service
from app.services import verify as verify_service
from app.schemas import HealthResponse

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_models()
    classify_service.init()
    verify_service.init()
    yield


app = FastAPI(title="PS-23 Civic Complaint ML Service", version="0.1.0", lifespan=lifespan)
for r in (classify, duplicate, priority, verify, hotspots):
    app.include_router(r.router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", models_loaded=models.loaded)
