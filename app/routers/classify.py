"""Router: POST /ml/classify"""
from fastapi import APIRouter

from app.schemas import ClassifyRequest, ClassifyResponse
from app.services.classify import classify

router = APIRouter(prefix="/ml", tags=["classify"])


@router.post("/classify", response_model=ClassifyResponse)
def classify_endpoint(req: ClassifyRequest) -> ClassifyResponse:
    return classify(req)
