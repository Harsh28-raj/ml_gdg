# Progress
Current phase: 6 (done except real Verified pair)
## Phases
- [x] Phase 0 — Setup — done (structure, config, schemas, model loader, utils, dummy endpoints, docs; 7 smoke tests pass; live server: models_loaded=true, all endpoints OK)
- [x] Phase 1 — Classify — done (CLIP prompt-ensemble + keyword/MiniLM; 12 tests pass, 3 skip until images added; ~10 ms text, <0.7 s image incl. first-call warm-up)
- [x] Phase 2 — Duplicate — done (filter + MiniLM/CLIP sims + distance bonus, SQLite cache, batched CLIP; 13 tests pass, 1 skip; 11 uncached images 1.3 s, cached ~0 s)
- [x] Phase 3 — Verify — done (GPS/time/pHash/CLIP same-place + issue prompts; 13 tests pass, 3 skip until real images; ~0.5 s with 2 large images)
- [x] Phase 4 — Priority — done (rule-based, no models; 14 tests pass; taar gira → High, kachra → Low, dup count boosts)
- [x] Phase 5 — Hotspots — done (DBSCAN haversine/ball_tree; 7 tests pass; point lat/lng made optional, additive)
- [x] Phase 6 — Polish — 10 real Commons photos added; thresholds + verify prompts tuned; README, scripts/demo.py (5/6 demo cases pass), scripts/tune.py, API.md checked. 77 tests pass, 2 skip (Verified pair).
## Known issues
- No real before/after pair of the same spot → demo case 4 (Verified) + 2 tests skip. Need pothole1_fixed.jpg / garbage1_clean.jpg (see sample_data/README.md).
- streetlight1.jpg is a weak sample (intact lamp on waterfront → classifies `other`); streetlight category untested on a real broken light.
- MiniLM is English-only: Hinglish↔English paraphrase text_sim ≈ 0.1, Hinglish↔Hinglish ≈ 0.65 (borderline). Image sim carries those cases.
- Python 3.14 venv (CLAUDE.md says 3.10+, OK). HF symlink warning on Windows is harmless.
## Environment (for new sessions)
- Repo root = ml-service. venv at `.venv` (Python 3.14, torch 2.14 CPU, transformers 5.20, sentence-transformers 6.2) — already installed, do NOT reinstall.
- Models already cached in `~/.cache/huggingface` (CLIP ViT-B/32 + MiniLM). Loaded via `app/models_loader.py` → `models.clip_model`, `models.clip_processor`, `models.text_model`.
- Run: `.venv/Scripts/python -m uvicorn app.main:app --port 8001` (log → server.log). Check port 8001 is free first.
- Tests: `.venv/Scripts/python -m pytest tests/...` (`ML_LOAD_MODELS=0` for smoke-only runs; conftest has session-scoped `client`).
- All 5 services implemented (no dummies left). Scripts: `scripts/demo.py` (needs running server), `scripts/tune.py` (in-process).
- Shared embedding helpers in `app/models_loader.py`: `clip_image_embedding(s)` (batched), `clip_text_embeddings`, `clip_logit_scale`, `text_embeddings` (all L2-normalized numpy) — reuse in Duplicate/Verify.
## Next step
- Add a real same-spot before/after pair → `python scripts/tune.py` (check same_place ≥ 0.75 and issue drop) → `pytest` → `python scripts/demo.py`.
- Optional: replace streetlight1.jpg with a clearly broken streetlight.
