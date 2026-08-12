"""Offline, dependency-light embedding stand-in.

Nothing is downloaded and no model runs: text becomes a signed feature-hash of its
words and character trigrams, images become a hash of colour/spatial/edge statistics
computed with Pillow. Both are deterministic, so the same input always yields the
same vector and the whole pipeline (ingest -> index -> ANN -> re-rank) can be
exercised and unit-tested with no API key.

What it is good for:
  * text -> text  search (lexical-ish similarity, not true semantics)
  * image -> image search (genuinely useful: colour and layout similarity)

What it cannot do:
  * text -> image search. The two live in unrelated subspaces here. Set
    EMBEDDING_PROVIDER=jina or =cohere for real cross-modal retrieval.
"""

import hashlib
import io
import itertools
import re
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from PIL import Image, ImageFilter

from app.embeddings.base import EmbeddingProvider, InputKind, Vector, l2_normalize

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Distinct salts keep the text and image feature families from colliding.
_TEXT_SALT = b"text:"
_IMAGE_SALT = b"image:"


def _bucket(token: bytes, dim: int) -> tuple[int, float]:
    """Map a feature name to (index, sign) using a stable hash."""
    digest = hashlib.blake2b(token, digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


class LocalHashProvider(EmbeddingProvider):
    name = "local_hash"
    supports_cross_modal = False

    async def embed_texts(self, texts: Sequence[str], kind: InputKind = "document") -> list[Vector]:
        return self._validate_dim([self._embed_text_sync(t) for t in texts])

    async def embed_images(
        self, images: Sequence[bytes], kind: InputKind = "document"
    ) -> list[Vector]:
        return self._validate_dim([self._embed_image_sync(b) for b in images])

    # --- text ---------------------------------------------------------------
    def _embed_text_sync(self, text: str) -> Vector:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())

        for token in tokens:
            idx, sign = _bucket(_TEXT_SALT + token.encode(), self.dim)
            vec[idx] += sign

        # Character trigrams add robustness to plurals and small typos.
        padded = f" {' '.join(tokens)} "
        for i in range(len(padded) - 2):
            idx, sign = _bucket(_TEXT_SALT + padded[i : i + 3].encode(), self.dim)
            vec[idx] += sign * 0.5

        # Bigrams carry a little word order.
        for a, b in itertools.pairwise(tokens):
            idx, sign = _bucket(_TEXT_SALT + f"{a}_{b}".encode(), self.dim)
            vec[idx] += sign * 0.75

        return l2_normalize(vec)

    # --- image --------------------------------------------------------------
    #: Relative pull of each feature family on the final image vector.
    _FAMILY_WEIGHTS: ClassVar[dict[str, float]] = {"grid": 1.0, "hsv": 0.9, "edge": 0.7}

    def _embed_image_sync(self, payload: bytes) -> Vector:
        with Image.open(io.BytesIO(payload)) as img:
            rgb = img.convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
            edges = rgb.convert("L").filter(ImageFilter.FIND_EDGES)
            hsv = rgb.convert("HSV")
            families = {
                "grid": self._grid_features(rgb),
                "hsv": self._histogram_features(hsv, "hsv"),
                "edge": self._histogram_features(edges, "edge"),
            }

        # Each family is hashed and normalised on its own before being combined, so a
        # family with a naturally larger magnitude cannot drown out the others.
        vec = np.zeros(self.dim, dtype=np.float32)
        for family, features in families.items():
            projected = np.zeros(self.dim, dtype=np.float32)
            for name, value in features.items():
                idx, sign = _bucket(_IMAGE_SALT + name.encode(), self.dim)
                projected[idx] += sign * float(value)
            norm = float(np.linalg.norm(projected))
            if norm > 0.0:
                vec += (projected / norm) * self._FAMILY_WEIGHTS[family]
        return l2_normalize(vec)

    @staticmethod
    def _grid_features(img: Image.Image) -> dict[str, float]:
        """Per-cell colour *deviation from the image mean* over an 8x8 grid.

        Storing raw cell means would make every image look alike: product shots are
        mostly background, so the shared backdrop dominates the vector and cosine
        similarity saturates near 1. Centring on the image's own mean throws away that
        common component and leaves the subject — where it sits, and how its colour
        departs from the backdrop — which is the part worth matching on.
        """
        arr = np.asarray(img, dtype=np.float32) / 255.0
        overall = arr.reshape(-1, 3).mean(axis=0)
        cells = 8
        step = arr.shape[0] // cells
        out: dict[str, float] = {}
        for row in range(cells):
            for col in range(cells):
                block = arr[row * step : (row + 1) * step, col * step : (col + 1) * step]
                deviation = block.reshape(-1, 3).mean(axis=0) - overall
                for channel, value in enumerate(deviation):
                    out[f"grid_{row}_{col}_c{channel}"] = float(value)
        return out

    @staticmethod
    def _histogram_features(img: Image.Image, prefix: str, bins: int = 24) -> dict[str, float]:
        """Channel histograms, mean-centred so the flat/uniform component drops out."""
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        out: dict[str, float] = {}
        for channel in range(arr.shape[2]):
            hist, _ = np.histogram(arr[:, :, channel], bins=bins, range=(0, 256))
            fractions = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
            centred = fractions - fractions.mean()
            for b, value in enumerate(centred):
                out[f"{prefix}_c{channel}_b{b}"] = float(value)
        return out
