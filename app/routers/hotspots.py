"""Router: POST /ml/hotspots"""
from fastapi import APIRouter

from app.schemas import HotspotsRequest, HotspotsResponse
from app.services.hotspots import find_hotspots

router = APIRouter(prefix="/ml", tags=["hotspots"])


@router.post("/hotspots", response_model=HotspotsResponse)
def hotspots_endpoint(req: HotspotsRequest) -> HotspotsResponse:
    return find_hotspots(req)
