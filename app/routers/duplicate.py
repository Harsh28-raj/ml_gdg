"""Router: POST /ml/duplicate"""
from fastapi import APIRouter

from app.schemas import DuplicateRequest, DuplicateResponse
from app.services.duplicate import find_duplicate

router = APIRouter(prefix="/ml", tags=["duplicate"])


@router.post("/duplicate", response_model=DuplicateResponse)
def duplicate_endpoint(req: DuplicateRequest) -> DuplicateResponse:
    return find_duplicate(req)
