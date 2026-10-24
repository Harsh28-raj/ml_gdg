"""Load a PIL image from http(s) URL, local path, or base64. Returns None on any failure."""
import base64
import binascii
import io
import logging
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

from app import config

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _open(data: bytes) -> Optional[Image.Image]:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except Exception as e:
        log.warning("Cannot decode image: %s", e)
        return None


def load_image(image_url: Optional[str] = None, image_base64: Optional[str] = None) -> Optional[Image.Image]:
    """Return an RGB PIL image, or None if missing/invalid. Base64 wins if both given."""
    if image_base64:
        b64 = image_base64.split(",", 1)[1] if image_base64.startswith("data:") else image_base64
        try:
            return _open(base64.b64decode(b64, validate=False))
        except (binascii.Error, ValueError) as e:
            log.warning("Bad base64 image: %s", e)
            return None

    if not image_url:
        return None

    if image_url.startswith(("http://", "https://")):
        try:
            r = requests.get(image_url, timeout=config.HTTP_TIMEOUT_S)
            r.raise_for_status()
            if len(r.content) > config.MAX_IMAGE_BYTES:
                log.warning("Image too large: %s", image_url)
                return None
            return _open(r.content)
        except requests.RequestException as e:
            log.warning("Image download failed %s: %s", image_url, e)
            return None

    path = Path(image_url)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        return _open(path.read_bytes())
    except OSError as e:
        log.warning("Cannot read image file %s: %s", path, e)
        return None
