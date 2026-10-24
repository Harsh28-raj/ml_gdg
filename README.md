# PS-23 Civic Complaint — ML Service

Zero-shot ML microservice (CLIP + MiniLM, no training) for categorizing complaints, detecting duplicates, scoring priority, verifying before/after resolution photos, and finding hotspots.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt   # pulls CPU-only torch via --extra-index-url
cp .env.example .env              # optional
```

## Run
```bash
uvicorn app.main:app --port 8001
```
First start downloads models (~600 MB) from Hugging Face. Set `ML_LOAD_MODELS=0` to run without models.

Open http://localhost:8001/docs.

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | /health | liveness + models_loaded |
| POST | /ml/classify | category from text and/or image |
| POST | /ml/duplicate | match against nearby candidates |
| POST | /ml/priority | rule-based priority |
| POST | /ml/verify | before/after resolution proof |
| POST | /ml/hotspots | DBSCAN clusters |

Full contract: [docs/API.md](docs/API.md). Sample images: [sample_data/README.md](sample_data/README.md).

## How it works (no training)
| Endpoint | Method |
|---|---|
| classify | CLIP zero-shot on image (3 prompts/category) · keywords (EN + Hinglish) + MiniLM on text · 0.6 image / 0.4 text |
| duplicate | same category, ≤ 100 m, ≤ 7 days → 0.5 MiniLM text sim + 0.5 CLIP image sim + distance bonus; ≥ 0.80 duplicate, 0.6–0.80 review |
| priority | rule-based: category base + urgent keywords (taar, current, aag, bacche, manhole…) + report count |
| verify | GPS ≤ 50 m, time order, pHash copy check, CLIP same-place ≥ 0.75, CLIP "issue present vs fixed" |
| hotspots | DBSCAN, haversine metric |

All thresholds/keywords/prompts live in `app/config.py`. Choices and measured values: [docs/DECISIONS.md](docs/DECISIONS.md).
Embeddings are cached in SQLite (`ML_CACHE_DB`, default `embedding_cache.sqlite3`) keyed by complaint_id.

## Demo
```bash
uvicorn app.main:app --port 8001          # terminal 1
python scripts/demo.py                    # terminal 2 (--base to change URL)
```
| # | Case | Expected |
|---|---|---|
| 1 | Same pothole reported twice, 24 m apart | `is_duplicate: true`, group G12 |
| 2 | Garbage photo, empty text (+ "Gali mein kachra pada hai") | `waste` |
| 3 | "Bijli ka taar gira hai" | `High` |
| 4 | pothole1 → pothole1_fixed | `Verified` |
| 5 | Same photo as before and after | `Rejected` |
| 6 | pothole1 → clean_road_other | `Suspicious` |

Cases needing images are skipped until the files in [sample_data/README.md](sample_data/README.md) are added.

## Tuning
After adding sample images: `python scripts/tune.py` prints CLIP category scores, pairwise CLIP cosine + pHash distance, and issue probabilities, to set `SAME_PLACE_MIN`, `ISSUE_*`, `DUP_THRESHOLD` in `app/config.py`.

## Tests
```bash
pytest                    # all (≈20 s, loads models once)
ML_LOAD_MODELS=0 pytest tests/test_smoke.py tests/test_priority.py tests/test_hotspots.py   # no models
```
Image-dependent tests skip when sample images are missing.

## Deploy on Render
1. Render dashboard → **New → Blueprint** → select this repo. `render.yaml` sets everything up.
2. Plan: **Standard (2 GB) or bigger** — the service uses ~620 MB steady, ~1 GB peak while loading models. Free/Starter (512 MB) will crash with out-of-memory.
3. Build downloads the models (~600 MB) once (`scripts/download_models.py`); first deploy takes several minutes.
4. Check `https://<your-service>.onrender.com/health` → `{"status":"ok","models_loaded":true}`, docs at `/docs`.
5. Give the backend team the base URL. On Render, send images as `http(s)` URLs or base64 (local paths only work for files inside this repo, e.g. `sample_data/images/...`).

Notes: one worker only (each worker loads its own copy of the models). The SQLite embedding cache lives on Render's ephemeral disk and resets on each deploy — that's fine, it's only a speed cache.
