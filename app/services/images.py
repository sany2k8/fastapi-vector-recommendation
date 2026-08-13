"""Image validation and on-disk storage.

Uploaded bytes are never trusted: Pillow must be able to decode them, the format
must be one the embedding providers accept, and the size is capped before anything
reaches an external API.
"""

import io
import uuid
from pathlib import Path

from PIL import Image

from app.core.config import get_settings
from app.core.errors import ValidationError

ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}


def validate_image(payload: bytes) -> str:
    """Return the detected Pillow format, raising ValidationError if unusable."""
    settings = get_settings()
    if not payload:
        raise ValidationError("uploaded image is empty")
    if len(payload) > settings.max_image_bytes:
        raise ValidationError(
            f"image is {len(payload)} bytes; the limit is {settings.max_image_bytes}"
        )
    try:
        with Image.open(io.BytesIO(payload)) as img:
            img.verify()  # structural check; consumes the file object
        with Image.open(io.BytesIO(payload)) as img:
            fmt = (img.format or "").upper()
    except Exception as exc:
        raise ValidationError(f"could not decode image: {exc}") from exc

    if fmt not in ALLOWED_FORMATS:
        raise ValidationError(
            f"unsupported image format {fmt or 'unknown'}; allowed: "
            f"{', '.join(sorted(ALLOWED_FORMATS))}"
        )
    return fmt


def normalize_image(payload: bytes) -> tuple[bytes, str]:
    """Downscale oversized images and return canonical (bytes, extension).

    Anything wider or taller than `MAX_IMAGE_DIMENSION` is resampled and re-encoded as
    JPEG. Beyond saving disk, this is a cost control: Voyage bills image embeddings by
    pixel, so importing 4000px product photos would burn quota for no retrieval gain.
    Images already within bounds are stored untouched.
    """
    fmt = validate_image(payload)
    limit = get_settings().max_image_dimension

    with Image.open(io.BytesIO(payload)) as img:
        if max(img.size) <= limit:
            return payload, ALLOWED_FORMATS[fmt]
        resized = img.convert("RGB")
        resized.thumbnail((limit, limit), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=88)
        return buffer.getvalue(), ".jpg"


def store_image(payload: bytes, product_id: uuid.UUID) -> tuple[str, bytes]:
    """Persist an image and return (filename, the bytes actually written).

    The caller must embed the *returned* bytes, not what it passed in. Re-indexing reads
    the stored file, so embedding the original while storing a normalized copy would make
    a product's vectors change silently on the next re-index.
    """
    canonical, suffix = normalize_image(payload)
    settings = get_settings()
    settings.image_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{product_id}{suffix}"
    (settings.image_dir / filename).write_bytes(canonical)
    return filename, canonical


def remove_image(filename: str | None) -> None:
    if not filename:
        return
    path = Path(get_settings().image_dir) / filename
    path.unlink(missing_ok=True)
