# CLAUDE.md — PS-23 Civic Complaint ML Service

> Read this file fully before doing anything. It is the single source of truth.
> Then read `docs/PROGRESS.md` to see what is already done. Never redo finished work.

---

## 1. Project Context

Hackathon problem **PS-23: Civic Complaint Tracking Platform**.
Citizens report waste, damaged roads, broken streetlights, water leaks. The platform must
categorize reports, group duplicates, assign to a simulated department, and track resolution with evidence.

**My scope = ONLY the ML service.** A separate main backend (FastAPI + DB) and a React frontend
are built by teammates. The main backend calls this ML service over HTTP.

```
Frontend → Main Backend (DB, users, status) → ML Service (this repo) → JSON response
```

Time limit: ~4 hours total. **No model training.** Only pretrained / zero-shot models.

---

## 2. Tech Stack (fixed — do not change without asking)

- Python 3.10+
- FastAPI + Uvicorn
- Pydantic v2 (request/response schemas)
- `open_clip_torch` or `transformers` CLIP → `openai/clip-vit-base-patch32`
- `sentence-transformers` → `all-MiniLM-L6-v2`
- `imagehash` + `Pillow` (pHash)
- `scikit-learn` (DBSCAN)
- `numpy`, `requests`
- `pytest` + `httpx` (tests)
- CPU-only PyTorch is fine

---

## 3. Folder Structure (create exactly this)

```
ml-service/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── app/
│   ├── main.py            # FastAPI app, startup model loading, routers
│   ├── config.py          # all thresholds/labels/constants (single place)
│   ├── models_loader.py   # loads CLIP + MiniLM ONCE, exposes singletons
│   ├── schemas.py         # all Pydantic request/response models
│   ├── utils/
│   │   ├── image_io.py    # load image from URL / local path / base64
│   │   ├── geo.py         # haversine distance (meters)
│   │   └── cache.py       # SQLite embedding cache keyed by complaint_id
│   ├── services/
│   │   ├── classify.py
│   │   ├── duplicate.py
│   │   ├── priority.py
│   │   ├── verify.py
│   │   └── hotspots.py
│   └── routers/
│       ├── classify.py
│       ├── duplicate.py
│       ├── priority.py
│       ├── verify.py
│       └── hotspots.py
├── tests/
│   ├── test_classify.py
│   ├── test_duplicate.py
│   ├── test_priority.py
│   ├── test_verify.py
│   └── test_hotspots.py
├── sample_data/
│   └── images/            # demo images (pothole, garbage, streetlight, clean road...)
└── docs/
    ├── PROGRESS.md        # phase checklist + what is done (UPDATE after every phase)
    ├── API.md             # final request/response contract for backend team
    └── DECISIONS.md       # short log of choices, thresholds, known limitations
```

---

## 4. Global Rules

1. **Work phase by phase** (Section 7). Finish a phase → run its tests → update docs → STOP and report to me in 3–5 lines. Then wait for "next".
2. Models load **once at startup** (FastAPI lifespan). Never load a model inside a request.
3. All thresholds, labels, keywords live in `app/config.py`. No magic numbers in services.
4. Every endpoint must **never crash** on missing photo / missing text / missing GPS / bad URL. Return a valid response with `needs_review: true` and a `reason`.
5. Services contain logic; routers only validate + call services.
6. Keep responses exactly as defined in Section 6. If you must change a field, update `docs/API.md` and tell me.
7. Use type hints and short docstrings. No over-engineering, no auth, no database other than the SQLite embedding cache.
8. Images can arrive as `image_url` (http/https or local path) or `image_base64`. Handle both in `utils/image_io.py`.
9. Target latency: < 2 s per request on CPU.

---

## 5. Token-Saving Rules (important)

- At the start of each session read ONLY: `CLAUDE.md` and `docs/PROGRESS.md`. Read other files only when the current phase needs them.
- Do not re-read or re-print entire files you just wrote. Edit only the parts that change.
- Do not paste large code blocks into chat explanations. Summaries only.
- Keep `docs/PROGRESS.md` short (checklist + one-line notes), not a diary.
- After each phase, update docs, then give a ≤5 line summary.

### docs/PROGRESS.md format
```
# Progress
Current phase: <N>
## Phases
- [x] Phase 0 — Setup — done (notes)
- [ ] Phase 1 — Classify
...
## Known issues
- ...
## Next step
- ...
```

### docs/API.md
For every endpoint: method, path, request JSON example, response JSON example, field meanings, error behaviour.

### docs/DECISIONS.md
One bullet per decision: what, why, value (e.g. "duplicate text threshold = 0.75 — tuned on sample_data").

---

## 6. API Contract

Base: `http://localhost:8001`

Common field in every response: `needs_review: bool`, `reason: str | null`.

### 6.1 GET /health
```json
{ "status": "ok", "models_loaded": true }
```

### 6.2 POST /ml/classify — category
Request:
```json
{ "complaint_id": "C1023", "text": "Gali mein kachra pada hai", "image_url": "sample_data/images/garbage1.jpg", "image_base64": null }
```
Response:
```json
{
  "complaint_id": "C1023",
  "category": "waste",
  "confidence": 0.95,
  "confidence_label": "High",
  "scores": { "waste": 0.95, "road": 0.02, "streetlight": 0.01, "water": 0.01, "other": 0.01 },
  "source": "text+image",
  "needs_review": false,
  "reason": null
}
```
Logic:
- Categories: `waste`, `road`, `streetlight`, `water`, `other`.
- Image score: CLIP zero-shot with 2–3 prompt variants per category (e.g. "a photo of a garbage pile on the street"), softmax.
- Text score: keyword match (English + Hindi/Hinglish, e.g. kachra, gaddha, khamba, light, paani, naali) combined with MiniLM similarity to category descriptions, softmax.
- Combine: both present → weighted average (image 0.6, text 0.4, in config). One present → use it. None → `other`, confidence 0, `source: "none"`, `needs_review: true`.
- confidence_label: High > 0.8, Medium 0.5–0.8, Low < 0.5. Low → `needs_review: true`.

### 6.3 POST /ml/duplicate — duplicate detection
Request (backend sends the new complaint + nearby candidates it already has):
```json
{
  "complaint": { "complaint_id": "C1050", "text": "...", "image_url": "...", "lat": 28.67, "lng": 77.45, "created_at": "2026-10-24T10:00:00", "category": "road" },
  "candidates": [
    { "complaint_id": "C1023", "group_id": "G12", "text": "...", "image_url": "...", "lat": 28.6702, "lng": 77.4501, "created_at": "2026-10-22T09:00:00", "category": "road" }
  ]
}
```
Response:
```json
{
  "complaint_id": "C1050",
  "is_duplicate": true,
  "matched_complaint_id": "C1023",
  "group_id": "G12",
  "similarity_score": 0.83,
  "details": { "text_sim": 0.79, "image_sim": 0.86, "distance_m": 24.5, "days_apart": 2 },
  "top_matches": [ { "complaint_id": "C1023", "score": 0.83, "distance_m": 24.5 } ],
  "needs_review": false,
  "reason": null
}
```
Logic:
- Filter candidates: same category, distance ≤ 100 m (haversine), within 7 days.
- Score = 0.5·text_sim + 0.5·image_sim (if one modality missing, use the other only).
- Distance bonus: closer = small boost (config).
- `is_duplicate` if score ≥ 0.75. Borderline 0.6–0.75 → `is_duplicate: false`, `needs_review: true`.
- Cache embeddings by complaint_id in SQLite (`utils/cache.py`) so candidates are not re-embedded.
- No candidates → `is_duplicate: false`, `group_id: null`.

### 6.4 POST /ml/priority — priority
Request:
```json
{ "complaint_id": "C1050", "text": "Bijli ka taar gira hai", "category": "streetlight", "duplicate_count": 5 }
```
Response:
```json
{
  "complaint_id": "C1050",
  "priority": "High",
  "priority_score": 0.9,
  "reasons": ["urgent keyword: taar", "5 people reported this"],
  "needs_review": false,
  "reason": null
}
```
Logic (rule-based, explainable):
- Urgent keywords (EN + Hindi/Hinglish: wire/taar, current, shock, accident, fire/aag, child/bacche, hospital, school, flood, open manhole, etc.) → big boost.
- Category base weight (config).
- duplicate_count boost (capped).
- score → High ≥ 0.7, Medium 0.4–0.7, Low < 0.4.

### 6.5 POST /ml/verify — before/after resolution proof
Request:
```json
{
  "complaint_id": "C1023",
  "category": "road",
  "before_image_url": "...", "after_image_url": "...",
  "complaint_lat": 28.67, "complaint_lng": 77.45,
  "after_lat": 28.6701, "after_lng": 77.4502,
  "complaint_created_at": "2026-10-22T09:00:00",
  "after_taken_at": "2026-10-24T12:00:00"
}
```
Response:
```json
{
  "complaint_id": "C1023",
  "status": "Verified",
  "verification_score": 0.87,
  "checks": {
    "location_ok": true, "distance_m": 18.2,
    "time_ok": true,
    "not_same_photo": true, "phash_distance": 34,
    "same_place_score": 0.84,
    "issue_before": 0.91, "issue_after": 0.08, "issue_removed": true
  },
  "reasons": [],
  "needs_review": false,
  "reason": null
}
```
Logic:
1. Location: after GPS within 50 m of complaint → `location_ok`. Missing GPS → not ok, add reason.
2. Time: after_taken_at > complaint_created_at.
3. Copy check: pHash distance between before/after; distance ≤ 5 → same photo → **Rejected**.
4. Same place: CLIP image embedding cosine similarity ≥ 0.70.
5. Issue removed: CLIP zero-shot "issue present" vs "issue fixed" prompts per category; before issue prob high AND after issue prob low (e.g. after < 0.4 and drop ≥ 0.3).
- Decision: location fail OR time fail OR same photo → `Rejected`. Same place AND issue removed → `Verified`. Otherwise → `Suspicious` with `needs_review: true`.
- verification_score = weighted combination (config).

### 6.6 POST /ml/hotspots — clusters (optional)
Request:
```json
{ "points": [ { "complaint_id": "C1", "lat": 28.67, "lng": 77.45, "category": "waste" } ], "eps_m": 200, "min_samples": 3 }
```
Response:
```json
{
  "clusters": [ { "cluster_id": 0, "center_lat": 28.671, "center_lng": 77.451, "count": 7, "top_category": "waste", "complaint_ids": ["C1","C4"] } ],
  "noise_count": 3,
  "needs_review": false,
  "reason": null
}
```
Logic: DBSCAN with haversine metric (radians), eps = eps_m / 6371000.

---

## 7. Phases (do strictly in order)

**Phase 0 — Setup**
- Create folder structure, requirements.txt, config.py, schemas.py (all schemas), models_loader.py, image_io, geo, cache, main.py with lifespan + /health.
- Create **dummy versions of all 5 endpoints** returning valid fixed responses matching Section 6 (so backend can integrate immediately).
- Create docs/PROGRESS.md, docs/API.md (from Section 6), docs/DECISIONS.md.
- Add a few sample images (or a script/readme telling me which images to put in `sample_data/images/`).
- Test: server starts, /health ok, /docs works, all dummy endpoints return valid JSON.

**Phase 1 — Classify** → replace dummy with real logic + tests (garbage image → waste, pothole → road, text-only, image-only, both empty).

**Phase 2 — Duplicate** → real logic + embedding cache + tests (same issue nearby → duplicate, far away → not, different category → not, no candidates).

**Phase 3 — Verify** → real logic + tests (good before/after → Verified, same photo → Rejected, far GPS → Rejected, unrelated clean photo → Suspicious).

**Phase 4 — Priority** → real logic + tests (taar gira → High, normal kachra → Medium/Low, high duplicate count boosts).

**Phase 5 — Hotspots** → real logic + tests.

**Phase 6 — Polish**
- Tune thresholds on sample_data, log values in DECISIONS.md.
- README: setup, run command (`uvicorn app.main:app --port 8001`), endpoint list, demo cases.
- `scripts/demo.py` that hits all endpoints with sample data and prints results.
- Final API.md check.

After each phase: run `pytest tests/test_<phase>.py`, update `docs/PROGRESS.md`, give ≤5 line summary, wait for "next".

---

## 8. Demo Cases (must pass by Phase 6)

1. Two complaints of the same pothole within 30 m → duplicate.
2. Garbage photo with empty text → `waste`.
3. "Bijli ka taar gira hai" → `High` priority.
4. Real before/after → `Verified`.
5. Same photo uploaded as before and after → `Rejected`.
6. Clean road photo from a different place → `Suspicious`.

---

## 9. Do NOT

- Do not train or fine-tune any model.
- Do not add auth, users, or a main database — that is the backend team's job.
- Do not change the API response shape silently.
- Do not load models per request.
- Do not use paid APIs.
