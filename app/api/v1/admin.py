"""Admin endpoints backing the UI panel: status, seed, re-index, clear.

Mutations return a job id immediately and run in the background; poll
`GET /admin/jobs/{id}` (or `GET /admin/status`) for progress.
"""

from fastapi import APIRouter, Depends, status

from app.api.deps import RegistryDep, SessionDep, require_admin
from app.core.errors import NotFoundError
from app.schemas.admin import (
    AdminStatus,
    ClearProviderRequest,
    ImportPresets,
    ImportPreview,
    JobRead,
    OfflineImportRequest,
    ReindexRequest,
    RemoteImportRequest,
    SeedRequest,
)
from app.seeding.importers import PRESETS
from app.services.admin import AdminJobs, AdminService, preview_remote

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/status", response_model=AdminStatus)
async def admin_status(session: SessionDep, registry: RegistryDep) -> AdminStatus:
    """Catalogue size, per-provider index coverage, and job history."""
    return await AdminService(session, registry).status()


@router.post("/seed", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
async def seed_catalog(payload: SeedRequest, registry: RegistryDep) -> JobRead:
    """Load the demo catalogue, optionally indexing it with several providers at once."""
    job = AdminJobs(registry).seed(
        reset=payload.reset,
        with_images=payload.with_images,
        providers=payload.providers,
    )
    return JobRead.model_validate(job)


@router.post("/reindex", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
async def reindex(payload: ReindexRequest, registry: RegistryDep) -> JobRead:
    """Rebuild vectors for the named providers, leaving the others' vectors intact."""
    job = AdminJobs(registry).reindex(
        providers=payload.providers, skip_existing=payload.skip_existing
    )
    return JobRead.model_validate(job)


@router.get("/import/presets", response_model=ImportPresets)
async def import_presets() -> ImportPresets:
    """Known remote feeds, with the field mapping each one needs."""
    return ImportPresets(presets=PRESETS)


@router.post("/import/offline", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
async def import_offline(payload: OfflineImportRequest, registry: RegistryDep) -> JobRead:
    """Add products supplied directly, with images drawn locally. No network involved."""
    return JobRead.model_validate(AdminJobs(registry).import_offline(payload))


@router.post("/import/remote/preview", response_model=ImportPreview)
async def import_remote_preview(payload: RemoteImportRequest, session: SessionDep) -> ImportPreview:
    """Dry run: fetch and map a feed, write nothing, and report what would land."""
    return await preview_remote(payload, session)


@router.post("/import/remote", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
async def import_remote(payload: RemoteImportRequest, registry: RegistryDep) -> JobRead:
    """Import a remote JSON product feed, downloading its photos."""
    return JobRead.model_validate(AdminJobs(registry).import_remote(payload))


@router.post("/clear-catalog", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
async def clear_catalog(registry: RegistryDep) -> JobRead:
    """Delete every product, vector, interaction and stored image."""
    return JobRead.model_validate(AdminJobs(registry).clear_catalog())


@router.post("/clear-provider", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
async def clear_provider(payload: ClearProviderRequest, registry: RegistryDep) -> JobRead:
    """Drop one provider's index while keeping the products and the other indexes."""
    return JobRead.model_validate(AdminJobs(registry).clear_provider(payload.provider))


@router.get("/jobs/{job_id}", response_model=JobRead)
async def get_job(job_id: str, registry: RegistryDep) -> JobRead:
    job = registry.get(job_id)
    if job is None:
        raise NotFoundError(f"job {job_id} not found")
    return JobRead.model_validate(job)
