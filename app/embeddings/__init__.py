"""Embedding providers.

Several can be live at once: the registry caches one instance per provider name so a
single process can embed with `local_hash` and query with `jina` in the same request
cycle. `EMBEDDING_PROVIDER` only decides the default when a caller does not name one.
"""

from typing import NamedTuple

from app.core.config import Settings, get_settings
from app.core.errors import ValidationError
from app.embeddings.base import (
    EmbeddingProvider,
    InputKind,
    Vector,
    l2_normalize,
    weighted_merge,
)
from app.embeddings.cohere import CohereProvider
from app.embeddings.gemini import GeminiProvider
from app.embeddings.jina import JinaProvider
from app.embeddings.local_hash import LocalHashProvider
from app.embeddings.voyage import VoyageProvider

__all__ = [
    "CohereProvider",
    "EmbeddingProvider",
    "GeminiProvider",
    "InputKind",
    "JinaProvider",
    "LocalHashProvider",
    "Vector",
    "VoyageProvider",
    "available_providers",
    "build_provider",
    "close_providers",
    "get_provider",
    "l2_normalize",
    "provider_capabilities",
    "provider_model_name",
    "weighted_merge",
]

_INSTANCES: dict[str, EmbeddingProvider] = {}

#: Capabilities are class attributes, so they can be read without an API key — the
#: admin UI needs to describe a provider it cannot yet instantiate.
_PROVIDER_CLASSES: dict[str, type[EmbeddingProvider]] = {
    "local_hash": LocalHashProvider,
    "jina": JinaProvider,
    "cohere": CohereProvider,
    "gemini": GeminiProvider,
    "voyage": VoyageProvider,
}


class Capabilities(NamedTuple):
    supports_images: bool
    supports_cross_modal: bool


def provider_capabilities(name: str) -> Capabilities:
    """What a provider can do, without needing credentials to find out."""
    cls = _PROVIDER_CLASSES.get(name)
    if cls is None:
        return Capabilities(supports_images=False, supports_cross_modal=False)
    return Capabilities(cls.supports_images, cls.supports_cross_modal)


def build_provider(name: str, settings: Settings) -> EmbeddingProvider:
    match name:
        case "jina":
            return JinaProvider(
                dim=settings.embedding_dim,
                api_key=settings.jina_api_key,
                model=settings.jina_model,
                timeout=settings.embedding_timeout_seconds,
                requests_per_minute=settings.provider_rpm.get(name, 0.0),
                image_batch_size=settings.provider_image_batch.get(name),
            )
        case "cohere":
            return CohereProvider(
                dim=settings.embedding_dim,
                api_key=settings.cohere_api_key,
                model=settings.cohere_model,
                timeout=settings.embedding_timeout_seconds,
                requests_per_minute=settings.provider_rpm.get(name, 0.0),
                image_batch_size=settings.provider_image_batch.get(name),
            )
        case "gemini":
            return GeminiProvider(
                dim=settings.embedding_dim,
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                timeout=settings.embedding_timeout_seconds,
                requests_per_minute=settings.provider_rpm.get(name, 0.0),
                image_batch_size=settings.provider_image_batch.get(name),
            )
        case "voyage":
            return VoyageProvider(
                dim=settings.embedding_dim,
                api_key=settings.voyage_api_key,
                model=settings.voyage_model,
                timeout=settings.embedding_timeout_seconds,
                requests_per_minute=settings.provider_rpm.get(name, 0.0),
                image_batch_size=settings.provider_image_batch.get(name),
            )
        case "local_hash":
            return LocalHashProvider(dim=settings.embedding_dim)

    raise ValidationError(f"unknown embedding provider {name!r}")


def get_provider(name: str | None = None) -> EmbeddingProvider:
    """Return a cached provider instance, defaulting to the configured one."""
    settings = get_settings()
    resolved = name or settings.embedding_provider

    if resolved not in _INSTANCES:
        if not settings.has_credentials(resolved):
            raise ValidationError(
                f"embedding provider {resolved!r} is not configured. "
                f"Available: {', '.join(settings.available_providers)}"
            )
        _INSTANCES[resolved] = build_provider(resolved, settings)
    return _INSTANCES[resolved]


def available_providers() -> list[str]:
    return get_settings().available_providers


def provider_model_name(name: str) -> str:
    """The concrete model id behind a provider, recorded with each stored vector."""
    settings = get_settings()
    match name:
        case "jina":
            return settings.jina_model
        case "cohere":
            return settings.cohere_model
        case "gemini":
            return settings.gemini_model
        case "voyage":
            return settings.voyage_model
        case _:
            return "feature-hash"


async def close_providers() -> None:
    for provider in _INSTANCES.values():
        await provider.aclose()
    _INSTANCES.clear()
