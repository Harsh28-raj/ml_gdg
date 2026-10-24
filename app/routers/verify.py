"""Router: POST /ml/verify"""
from fastapi import APIRouter

from app.schemas import VerifyRequest, VerifyResponse
from app.services.verify import verify

router = APIRouter(prefix="/ml", tags=["verify"])


@router.post("/verify", response_model=VerifyResponse)
def verify_endpoint(req: VerifyRequest) -> VerifyResponse:
    return verify(req)
