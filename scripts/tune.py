"""Print raw model values on sample_data/images to tune thresholds in app/config.py.

Usage: python scripts/tune.py   (runs in-process, no server needed)
"""
import sys
from itertools import combinations
from pathlib import Path

import imagehash

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config  # noqa: E402
from app.models_loader import clip_image_embeddings, load_models  # noqa: E402
from app.services import classify, verify  # noqa: E402
from app.utils.image_io import load_image  # noqa: E402

IMG = Path(__file__).resolve().parents[1] / "sample_data" / "images"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console
    files = sorted(p for p in IMG.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if not files:
        sys.exit(f"No images in {IMG} — see sample_data/README.md")
    load_models()
    classify.init()
    verify.init()
    imgs = {p.name: load_image(str(p)) for p in files}
    embs = dict(zip(imgs, clip_image_embeddings(list(imgs.values()))))

    print("== Classify (image only) ==")
    for n, im in imgs.items():
        s = classify.image_scores(im)
        top = sorted(zip(config.CATEGORIES, s), key=lambda x: -x[1])[:2]
        print(f"  {n:28s} " + "  ".join(f"{c}={p:.2f}" for c, p in top))

    print(f"\n== Pairwise: CLIP cosine (same-place / dup) | pHash distance ==  SAME_PLACE_MIN={config.SAME_PLACE_MIN}")
    for a, b in combinations(imgs, 2):
        cos = float(embs[a] @ embs[b])
        ph = imagehash.phash(imgs[a]) - imagehash.phash(imgs[b])
        print(f"  {a:24s} {b:24s} cos={cos:.3f}  phash={ph}")

    print(f"\n== Issue present prob per category ==  after<{config.ISSUE_AFTER_MAX}, drop>={config.ISSUE_DROP_MIN}")
    for n in imgs:
        print(f"  {n:28s} " + "  ".join(f"{c}={verify._issue_prob(embs[n], c):.2f}" for c in config.VERIFY_PROMPTS))


if __name__ == "__main__":
    main()
