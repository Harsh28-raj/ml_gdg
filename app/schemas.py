"""All Pydantic request/response models (API contract, CLAUDE.md Section 6)."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    needs_review: bool = False
    reason: Optional[str] = None


# ---------- Health ----------
class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


# ---------- Classify ----------
class ClassifyRequest(BaseModel):
    complaint_id: str
    text: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None


class ClassifyResponse(BaseResponse):
    complaint_id: str
    category: str
    confidence: float
    confidence_label: str
    scores: dict[str, float]
    source: str  # "text+image" | "text" | "image" | "none"


# ---------- Duplicate ----------
class ComplaintItem(BaseModel):
    complaint_id: str
    group_id: Optional[str] = None
    text: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    created_at: Optional[datetime] = None
    category: Optional[str] = None


class DuplicateRequest(BaseModel):
    complaint: ComplaintItem
    candidates: list[ComplaintItem] = Field(default_factory=list)


class DuplicateDetails(BaseModel):
    text_sim: Optional[float] = None
    image_sim: Optional[float] = None
    distance_m: Optional[float] = None
    days_apart: Optional[float] = None


class TopMatch(BaseModel):
    complaint_id: str
    score: float
    distance_m: Optional[float] = None


class DuplicateResponse(BaseResponse):
    complaint_id: str
    is_duplicate: bool
    matched_complaint_id: Optional[str] = None
    group_id: Optional[str] = None
    similarity_score: float = 0.0
    details: Optional[DuplicateDetails] = None
    top_matches: list[TopMatch] = Field(default_factory=list)


# ---------- Priority ----------
class PriorityRequest(BaseModel):
    complaint_id: str
    text: Optional[str] = None
    category: Optional[str] = None
    duplicate_count: int = 0


class PriorityResponse(BaseResponse):
    complaint_id: str
    priority: str  # High | Medium | Low
    priority_score: float
    reasons: list[str] = Field(default_factory=list)


# ---------- Verify ----------
class VerifyRequest(BaseModel):
    complaint_id: str
    category: Optional[str] = None
    before_image_url: Optional[str] = None
    before_image_base64: Optional[str] = None
    after_image_url: Optional[str] = None
    after_image_base64: Optional[str] = None
    complaint_lat: Optional[float] = None
    complaint_lng: Optional[float] = None
    after_lat: Optional[float] = None
    after_lng: Optional[float] = None
    complaint_created_at: Optional[datetime] = None
    after_taken_at: Optional[datetime] = None


class VerifyChecks(BaseModel):
    location_ok: bool = False
    distance_m: Optional[float] = None
    time_ok: bool = False
    not_same_photo: Optional[bool] = None
    phash_distance: Optional[int] = None
    same_place_score: Optional[float] = None
    issue_before: Optional[float] = None
    issue_after: Optional[float] = None
    issue_removed: Optional[bool] = None


class VerifyResponse(BaseResponse):
    complaint_id: str
    status: str  # Verified | Rejected | Suspicious
    verification_score: float
    checks: VerifyChecks
    reasons: list[str] = Field(default_factory=list)


# ---------- Hotspots ----------
class HotspotPoint(BaseModel):
    complaint_id: str
    lat: Optional[float] = None  # missing/invalid GPS -> point skipped (reason set)
    lng: Optional[float] = None
    category: Optional[str] = None


class HotspotsRequest(BaseModel):
    points: list[HotspotPoint] = Field(default_factory=list)
    eps_m: float = 200.0
    min_samples: int = 3


class Cluster(BaseModel):
    cluster_id: int
    center_lat: float
    center_lng: float
    count: int
    top_category: Optional[str] = None
    complaint_ids: list[str]


class HotspotsResponse(BaseResponse):
    clusters: list[Cluster] = Field(default_factory=list)
    noise_count: int = 0
