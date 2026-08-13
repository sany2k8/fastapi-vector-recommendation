"""Admin (seed / re-index / clear) request and job schemas."""

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.importing import FieldMapping

JobKind = Literal[
    "seed", "reindex", "clear_catalog", "clear_provider", "import_offline", "import_remote"
]
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


class OfflineProduct(BaseModel):
    """One product supplied directly by the admin, with no network involved."""

    sku: str | None = Field(default=None, description="Generated from the name if omitted.")
    name: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=4000)
    category: str = Field(default="uncategorised", max_length=120)
    brand: str | None = Field(default=None, max_length=120)
    price: Decimal = Field(default=Decimal("0.00"), ge=0)
    color: str | None = Field(
        default=None,
        description="Drives the generated image. Unknown words get a stable derived colour.",
    )
    attributes: dict[str, str] = Field(default_factory=dict)


class OfflineImportRequest(BaseModel):
    products: list[OfflineProduct] = Field(min_length=1)
    generate_images: bool = Field(
        default=True, description="Draw a product image locally for each item."
    )
    providers: list[str] = Field(default_factory=list)
    sku_prefix: str = Field(default="GEN", max_length=16)


class RemoteImportRequest(BaseModel):
    url: str | None = Field(default=None, description="JSON feed to import. Ignored if preset set.")
    preset: str | None = Field(default=None, description="A known feed, e.g. 'dummyjson'.")
    mapping: FieldMapping | None = Field(
        default=None, description="Field mapping. Defaults to the preset's, else DummyJSON's."
    )
    limit: int = Field(default=50, ge=1, le=500)
    download_images: bool = Field(default=True)
    providers: list[str] = Field(default_factory=list)
    sku_prefix: str = Field(
        default="", max_length=16, description="Derived from the host if empty."
    )


class ImportPreview(BaseModel):
    """What a remote import *would* bring in, without writing anything."""

    source: str
    total_available: int
    sample: list[dict[str, object]]
    with_images: int
    already_present: int


class ImportPresets(BaseModel):
    presets: dict[str, dict[str, object]]


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
