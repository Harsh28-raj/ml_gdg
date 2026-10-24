"""All thresholds, labels, keywords and constants live here (single source)."""
import os

from dotenv import load_dotenv

load_dotenv()

# ---------- Runtime ----------
LOAD_MODELS: bool = os.getenv("ML_LOAD_MODELS", "1") == "1"
DEVICE: str = os.getenv("ML_DEVICE", "cpu")
CACHE_DB_PATH: str = os.getenv("ML_CACHE_DB", "embedding_cache.sqlite3")
HTTP_TIMEOUT_S: float = float(os.getenv("ML_HTTP_TIMEOUT", "5"))
MAX_IMAGE_BYTES: int = 10 * 1024 * 1024

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- Classify ----------
CATEGORIES = ["waste", "road", "streetlight", "water", "other"]

CLIP_PROMPTS: dict[str, list[str]] = {
    "waste": [
        "a photo of a garbage pile on the street",
        "a photo of overflowing trash bins",
        "a photo of litter and waste dumped on the roadside",
    ],
    "road": [
        "a photo of a pothole on a road",
        "a photo of a damaged broken road",
        "a photo of cracked asphalt with holes",
    ],
    "streetlight": [
        "a photo of a broken street light",
        "a photo of a street lamp pole",
        "a photo of a dark street with a non-working streetlight",
    ],
    "water": [
        "a photo of a water leak from a pipe",
        "a photo of a waterlogged flooded street",
        "a photo of an overflowing drain or sewage",
    ],
    "other": [
        "a photo of a city street",
        "a photo of a building",
        "a photo of people",
    ],
}

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "waste": "garbage, trash, litter, waste dumped, dustbin overflowing, kachra",
    "road": "pothole, damaged road, broken road, cracks in road, gaddha, sadak tooti",
    "streetlight": "street light not working, broken lamp, electric pole, bijli, khamba",
    "water": "water leakage, pipe burst, drain overflow, sewage, waterlogging, paani, naali",
    "other": "general civic complaint, noise, encroachment, stray animals",
}

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "waste": ["garbage", "trash", "litter", "waste", "dustbin", "dump", "kachra", "kooda", "kuda", "gandagi"],
    "road": ["pothole", "road", "crack", "gaddha", "gadda", "sadak", "footpath", "tooti sadak"],
    "streetlight": ["streetlight", "street light", "lamp", "light", "bulb", "pole", "khamba", "bijli", "andhera"],
    "water": ["water", "leak", "pipe", "drain", "sewage", "flood", "paani", "pani", "naali", "nali", "jal board"],
    "other": [],
}

IMAGE_WEIGHT = 0.6
TEXT_WEIGHT = 0.4
KEYWORD_WEIGHT = 0.6          # within text score: keyword vs MiniLM similarity
SEMANTIC_WEIGHT = 0.4
SOFTMAX_TEMPERATURE_TEXT = 0.1   # text raw scores in [0,1] -> sharpen
# Keyword = whole word, optionally followed by one of these suffixes (kachre, potholes, leaking...)
KEYWORD_SUFFIXES = ["s", "es", "e", "i", "ing", "ed", "en", "on", "iyan"]

CONF_HIGH = 0.8
CONF_MEDIUM = 0.5

# ---------- Duplicate ----------
DUP_MAX_DISTANCE_M = 100.0
DUP_MAX_DAYS = 7
DUP_TEXT_WEIGHT = 0.5
DUP_IMAGE_WEIGHT = 0.5
DUP_DISTANCE_BONUS_MAX = 0.05  # linear bonus: full at 0 m, 0 at DUP_MAX_DISTANCE_M
DUP_THRESHOLD = 0.80          # tuned: different places reach CLIP cos 0.72 (+0.05 bonus); same spot 0.86-0.98
DUP_BORDERLINE = 0.60
DUP_TOP_K = 3

# ---------- Priority ----------
URGENT_KEYWORDS: list[str] = [
    "wire", "taar", "current", "shock", "electrocuted", "electrocution", "spark", "accident", "fire", "aag",
    "child", "children", "bacche", "baccha", "hospital", "school", "flood", "open manhole",
    "manhole", "gas leak", "collapse", "injured", "ghayal", "khatra", "danger",
]
URGENT_BOOST = 0.4            # first urgent keyword
URGENT_EXTRA_PER = 0.05       # each additional distinct urgent keyword
URGENT_BOOST_CAP = 0.5
# English-only suffixes for urgent keywords (shared KEYWORD_SUFFIXES would make "aag" match "aage" = ahead)
URGENT_KEYWORD_SUFFIXES = ["s", "es", "ed", "ing"]
CATEGORY_BASE_WEIGHT: dict[str, float] = {
    "streetlight": 0.35, "water": 0.35, "road": 0.3, "waste": 0.25, "other": 0.2,
}
DUP_COUNT_BOOST_PER = 0.05
DUP_COUNT_BOOST_CAP = 0.25
PRIORITY_HIGH = 0.7
PRIORITY_MEDIUM = 0.4

# ---------- Verify ----------
VERIFY_MAX_DISTANCE_M = 50.0
PHASH_SAME_PHOTO_MAX = 5
SAME_PLACE_MIN = 0.75         # tuned: different places 0.46-0.72, same place 0.86-0.98 (margin for scene change after fix)
ISSUE_AFTER_MAX = 0.4
ISSUE_DROP_MIN = 0.3
VERIFY_WEIGHTS: dict[str, float] = {
    "location": 0.15, "time": 0.05, "not_same_photo": 0.1, "same_place": 0.3, "issue_removed": 0.4,
}

VERIFY_PROMPTS: dict[str, dict[str, list[str]]] = {
    "waste": {
        "present": ["a photo of a garbage pile on the street", "a photo of trash and litter on the ground"],
        "fixed": ["a photo of a clean street with no garbage", "a photo of a swept clean roadside"],
    },
    "road": {  # tuned: old fixed prompts ("freshly paved asphalt") gave a clean highway issue=0.79, now 0.03
        "present": ["a close-up photo of a pothole in the road", "a photo of a broken road with potholes",
                    "a photo of a damaged road surface with cracks and holes"],
        "fixed": ["a photo of a smooth road with no potholes", "a photo of a newly repaired road surface",
                  "a photo of a clean, well-maintained road"],
    },
    "streetlight": {  # tuned: daytime photos can't show "lit"; describe physical damage instead (clean road 0.69 -> 0.04)
        "present": ["a photo of a broken street light with smashed lamp", "a photo of a damaged, bent or fallen lamp post",
                    "a photo of a street light with exposed hanging wires"],
        "fixed": ["a photo of an intact street light in good condition", "a photo of a working street lamp",
                  "a photo of a street with normal street lights"],
    },
    "water": {
        "present": ["a photo of a leaking pipe or waterlogged street", "a photo of an overflowing drain"],
        "fixed": ["a photo of a dry street with no water leak", "a photo of a clean repaired drain"],
    },
    "other": {
        "present": ["a photo of a civic problem on the street"],
        "fixed": ["a photo of a clean tidy street"],
    },
}

# ---------- Hotspots ----------
EARTH_RADIUS_M = 6371000.0
HOTSPOT_DEFAULT_EPS_M = 200.0
HOTSPOT_DEFAULT_MIN_SAMPLES = 3
