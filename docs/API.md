# ML Service API

Base URL: `http://localhost:8001` · Interactive docs: `/docs`

Every `/ml/*` response contains:
- `needs_review: bool` — true when the result is uncertain or input was incomplete; a human should check.
- `reason: str | null` — why `needs_review` is set (or other note).

**Error behaviour (all endpoints):** missing photo / text / GPS or a bad image URL never crashes the service. You get HTTP 200 with a valid response, `needs_review: true` and a `reason`. HTTP 422 is returned only for malformed JSON or wrong field types (e.g. missing `complaint_id`).

Images: send `image_url` (http/https URL or a path relative to the ML service root) **or** `image_base64` (raw or `data:image/...;base64,` prefix). If both are sent, base64 is used.


---

## GET /health
```json
{ "status": "ok", "models_loaded": true }
```
`models_loaded` is false if models were skipped (`ML_LOAD_MODELS=0`) or failed to load.

---

## POST /ml/classify
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
| Field | Meaning |
|---|---|
| category | one of `waste`, `road`, `streetlight`, `water`, `other` |
| confidence | probability of the chosen category (0–1) |
| confidence_label | `High` > 0.8, `Medium` 0.5–0.8, `Low` < 0.5 (Low → needs_review) |
| scores | probability per category, sums to ~1 |
| source | `text+image`, `text`, `image`, or `none` |

No text and no usable image → `category: "other"`, `confidence: 0`, `source: "none"`, `needs_review: true`.
Image given but unreadable (bad URL/base64) → classified from text only, `needs_review: true`, `reason: "image could not be loaded"`.

---

## POST /ml/duplicate
Backend sends the new complaint plus nearby candidates it already has.

Request:
```json
{
  "complaint": { "complaint_id": "C1050", "text": "Sadak pe bada gaddha", "image_url": "sample_data/images/pothole1_b.jpg", "lat": 28.67, "lng": 77.45, "created_at": "2026-10-24T10:00:00", "category": "road" },
  "candidates": [
    { "complaint_id": "C1023", "group_id": "G12", "text": "Pothole near market", "image_url": "sample_data/images/pothole1.jpg", "lat": 28.6702, "lng": 77.4501, "created_at": "2026-10-22T09:00:00", "category": "road" }
  ]
}
```
Optional on each item: `image_base64`, `group_id`.

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
| Field | Meaning |
|---|---|
| is_duplicate | true if best score ≥ 0.80 |
| matched_complaint_id | best match if duplicate **or borderline** (so a reviewer can compare); else null |
| group_id | matched candidate's `group_id` if duplicate, else null. Duplicate but candidate had no group → null: backend creates the group |
| similarity_score | best combined score (0–1) |
| details | breakdown for best match; `text_sim`/`image_sim` null if modality missing |
| top_matches | up to 3 best eligible candidates |

Candidates are only considered if same category, ≤ 100 m, ≤ 7 days apart. Score 0.6–0.80 → `is_duplicate: false`, `needs_review: true`. No candidates → `is_duplicate: false`, `group_id: null`, `details: null`.
Missing GPS / `created_at` / `category` on either side → that filter is skipped for the pair, `needs_review: true` with a reason. Candidate with the same `complaint_id` as the new complaint is ignored.
Embeddings are cached by `complaint_id` (send stable IDs; text/image of an ID is assumed not to change).

---

## POST /ml/priority
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
| Field | Meaning |
|---|---|
| priority | `High` ≥ 0.7, `Medium` 0.4–0.7, `Low` < 0.4 |
| priority_score | 0–1 rule-based score |
| reasons | human-readable explanation of each boost |

Score = category base (streetlight/water 0.35, road 0.3, waste 0.25, other 0.2) + urgent keyword boost (0.4 first, +0.05 each extra, max 0.5) + 0.05 × duplicate_count (max 0.25), capped at 1.
Missing text or unknown category → still scored, `needs_review: true` with a reason. Negative duplicate_count treated as 0.

---

## POST /ml/verify
Request:
```json
{
  "complaint_id": "C1023",
  "category": "road",
  "before_image_url": "sample_data/images/pothole1.jpg",
  "after_image_url": "sample_data/images/pothole1_fixed.jpg",
  "complaint_lat": 28.67, "complaint_lng": 77.45,
  "after_lat": 28.6701, "after_lng": 77.4502,
  "complaint_created_at": "2026-10-22T09:00:00",
  "after_taken_at": "2026-10-24T12:00:00"
}
```
Optional: `before_image_base64`, `after_image_base64`.

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
| Field | Meaning |
|---|---|
| status | `Verified`, `Rejected`, or `Suspicious` (Suspicious → needs_review) |
| verification_score | weighted combination of checks (0–1) |
| checks.location_ok | after-photo GPS within 50 m of complaint (missing GPS → false) |
| checks.time_ok | after_taken_at later than complaint_created_at |
| checks.not_same_photo / phash_distance | pHash distance ≤ 5 → same photo → Rejected |
| checks.same_place_score | CLIP cosine similarity before vs after (≥ 0.75 = same place) |
| checks.issue_before / issue_after | CLIP probability the issue is visible |
| checks.issue_removed | before high, after < 0.4 and drop ≥ 0.3 |
| reasons | list of failed checks |

Decision: GPS > 50 m OR after photo not later OR same photo → `Rejected`; same place AND issue removed (and GPS + time present and ok) → `Verified`; else `Suspicious`. **Missing** GPS or timestamps → `Suspicious` (not Rejected), since we can't prove fraud. Image checks are null if an image is missing/unreadable.
`needs_review` is true only for `Suspicious`. `reason` = first entry of `reasons` (null when Verified).

---

## POST /ml/hotspots
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
| Field | Meaning |
|---|---|
| clusters | DBSCAN (haversine) clusters with ≥ `min_samples` points within `eps_m` |
| top_category | most common category in the cluster |
| noise_count | points not in any cluster |

Empty `points` → `clusters: []`, `noise_count: 0`.
Clusters sorted by size (cluster_id 0 = largest); center = mean lat/lng; `top_category` null if no member has a category.
`lat`/`lng` per point are optional: missing or out-of-range points are skipped (not counted as noise), `needs_review: true`, reason says how many. `eps_m` ≤ 0 or `min_samples` < 1 → defaults (200 m, 3) + reason.
