"""Application settings. All configuration flows through here — no bare os.environ."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, get_args

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EmbeddingProviderName = Literal["local_hash", "jina", "cohere", "gemini", "voyage"]
ALL_PROVIDERS: tuple[EmbeddingProviderName, ...] = get_args(EmbeddingProviderName)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database -------------------------------------------------------------
    database_url: str = "postgresql+psycopg://recsys:recsys@localhost:5439/recsys"
    db_echo: bool = False

    # --- API ------------------------------------------------------------------
    api_port: int = 8800
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Admin endpoints seed, re-index and clear the catalogue. Fine locally; turn this
    # off (or put an auth proxy in front) anywhere the API is publicly reachable.
    admin_enabled: bool = True

    # --- Embeddings -----------------------------------------------------------
    #: The provider used when a request does not name one.
    embedding_provider: EmbeddingProviderName = "local_hash"

    #: Shared vector width across every provider, so one column type and one index
    #: definition serve all of them. 512 is valid for all three
    #: (Jina 1-2048 via Matryoshka, Cohere 256/512/1024/1536, local_hash arbitrary).
    embedding_dim: int = 512

    jina_api_key: str = ""
    jina_model: str = "jina-embeddings-v4"
    cohere_api_key: str = ""
    cohere_model: str = "embed-v4.0"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-embedding-001"
    voyage_api_key: str = ""
    #: voyage-multimodal-3.5 supports 256/512/1024/2048. The older voyage-multimodal-3
    #: is fixed at 1024, so selecting it also means setting EMBEDDING_DIM=1024.
    voyage_model: str = "voyage-multimodal-3.5"
    embedding_timeout_seconds: float = 30.0

    #: Client-side pacing, requests per minute, 0 = unlimited. Free tiers are strict:
    #: Voyage without a payment method allows only **3 RPM / 10K TPM**, so pacing here
    #: turns a guaranteed 429 into a wait. Raise these once you add billing.
    #: Voyage is set *below* its stated 3 RPM on purpose: pacing at exactly the limit
    #: sits on the boundary and still trips 429s, and each rejected attempt costs
    #: another interval. 2 RPM (30s apart) completes a 28-product re-index reliably.
    provider_rpm: dict[str, float] = Field(
        default_factory=lambda: {"jina": 60.0, "cohere": 60.0, "gemini": 100.0, "voyage": 2.0}
    )

    #: Images per embedding request, per provider. Measured against Voyage's free tier:
    #: 1 and 4 images succeed (468 and 1,872 tokens), 8 returns 429. Images are billed by
    #: pixel and dwarf text, so this is the knob that actually decides whether a
    #: rate-limited re-index completes.
    provider_image_batch: dict[str, int] = Field(default_factory=lambda: {"voyage": 4})

    #: Products embedded per request batch during a re-index. Batching is what keeps a
    #: 3-RPM provider usable: one request per batch instead of two per product.
    reindex_batch_size: int = 16

    # --- Ranking --------------------------------------------------------------
    default_text_weight: float = 0.6
    ann_candidate_multiplier: int = 8
    max_top_k: int = 100

    #: Drop hits scoring below this fraction of the best hit. Relative rather than
    #: absolute because similarity scales differ wildly between providers and
    #: modalities — see docs/ranking notes in the README.
    default_min_score_ratio: float = 0.45
    #: Optional hard floor on the fused score, applied to every provider. None disables it.
    default_min_score: float | None = None

    #: Per-provider absolute floors, used when a request does not set `min_score`.
    #:
    #: These exist because the relative cutoff alone is not enough. Providers differ
    #: hugely in how they spread scores: measured on this catalogue, Jina puts a strong
    #: match at ~0.60 and an unrelated product at ~0.50 — a band so tight that a ratio
    #: cutoff never fires — while Cohere spreads 0.33 down to 0.15. A floor calibrated
    #: per provider is the part that actually removes junk from a compressed ranking.
    #:
    #: Starting points, not laws: re-measure against your own catalogue. Setting a
    #: provider to 0 disables its floor.
    #: gemini/voyage floors are untested placeholders — measure before trusting them.
    provider_min_scores: dict[str, float] = Field(
        default_factory=lambda: {
            "local_hash": 0.06,
            "jina": 0.50,
            "cohere": 0.18,
            "gemini": 0.0,
            "voyage": 0.0,
        }
    )

    def min_score_for(self, provider: str) -> float | None:
        """The absolute floor to apply for a provider when the caller gave none."""
        if self.default_min_score is not None:
            return self.default_min_score
        floor = self.provider_min_scores.get(provider)
        return floor if floor else None

    # --- Storage --------------------------------------------------------------
    image_dir: Path = PROJECT_ROOT / "data" / "images"
    max_image_bytes: int = 5 * 1024 * 1024

    @model_validator(mode="after")
    def _check_configuration(self) -> "Settings":
        if not 1 <= self.embedding_dim <= 4096:
            raise ValueError("EMBEDDING_DIM must be between 1 and 4096")
        if self.embedding_provider not in self.available_providers:
            raise ValueError(
                f"EMBEDDING_PROVIDER={self.embedding_provider} is the default but has no "
                f"API key configured. Available: {', '.join(self.available_providers)}"
            )
        if not 0.0 <= self.default_min_score_ratio <= 1.0:
            raise ValueError("DEFAULT_MIN_SCORE_RATIO must be between 0 and 1")
        return self

    def has_credentials(self, provider: str) -> bool:
        """Whether this provider can actually be constructed with current settings."""
        match provider:
            case "local_hash":
                return True
            case "jina":
                return bool(self.jina_api_key)
            case "cohere":
                return bool(self.cohere_api_key)
            case "gemini":
                return bool(self.gemini_api_key)
            case "voyage":
                return bool(self.voyage_api_key)
        return False

    @property
    def available_providers(self) -> list[str]:
        """Providers the process can instantiate right now, default listed first."""
        usable: list[str] = [p for p in ALL_PROVIDERS if self.has_credentials(p)]
        if self.embedding_provider in usable:
            usable.remove(self.embedding_provider)
            usable.insert(0, self.embedding_provider)
        return usable


@lru_cache
def get_settings() -> Settings:
    return Settings()
