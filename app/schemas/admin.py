"""Admin (seed / re-index / clear) request and job schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

JobKind = Literal["seed", "reindex", "clear_catalog", "clear_provider"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class SeedRequest(BaseModel):
    reset: bool = Field(
        default=False, description="Delete every product and interaction before seeding."
    )
    with_images: bool = Field(default=True, description="Generate and embed product images.")
    providers: list[str] = Field(
        default_factory=list,
        description=(
            "Providers to index the seeded catalogue with. Empty means just the default "
            "one. Listing several builds several indexes so results can be compared."
        ),
    )


class ReindexRequest(BaseModel):
    providers: list[str] = Field(
        default_factory=list,
        description="Providers to (re)build. Empty means the default provider only.",
    )
    skip_existing: bool = Field(
        default=False,
        description=(
            "Only embed products this provider has not indexed yet. Use it to resume "
            "after a rate-limit failure instead of paying for the whole catalogue again."
        ),
    )


class ClearProviderRequest(BaseModel):
    provider: str = Field(description="Drop only this provider's vectors; products stay.")


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: JobKind
    status: JobStatus
    detail: str
    processed: int
    total: int
    started_at: datetime
    finished_at: datetime | None
    error: str | None


class ProviderStatus(BaseModel):
    name: str
    configured: bool = Field(description="Credentials present, so it can be used right now.")
    indexed: bool = Field(description="Has vectors stored in the database.")
    supports_images: bool = Field(description="False for text-only models like Gemini.")
    cross_modal: bool
    is_default: bool
    products: int
    with_images: int
    model: str


class AdminStatus(BaseModel):
    products: int
    products_with_images: int
    interactions: int
    embedding_dim: int
    default_text_weight: float
    default_min_score_ratio: float
    providers: list[ProviderStatus]
    active_job: JobRead | None
    recent_jobs: list[JobRead]
