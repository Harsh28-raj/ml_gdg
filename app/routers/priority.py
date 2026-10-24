"""Router: POST /ml/priority"""
from fastapi import APIRouter

from app.schemas import PriorityRequest, PriorityResponse
from app.services.priority import score_priority

router = APIRouter(prefix="/ml", tags=["priority"])


@router.post("/priority", response_model=PriorityResponse)
def priority_endpoint(req: PriorityRequest) -> PriorityResponse:
    return score_priority(req)
