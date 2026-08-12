"""A tiny in-process job runner for the admin actions.

Seeding and re-indexing can take a while — with a hosted provider each product costs
two HTTP round trips — so the endpoints return immediately with a job id and the UI
polls for progress. One job runs at a time: these actions rewrite shared state, and
letting a seed and a re-index interleave would produce a half-written index.

Deliberately in-memory: jobs do not survive a restart, and this does not work across
replicas. That is the right trade for an admin panel on a single-process demo; a real
deployment would put this on a task queue.
"""

import asyncio
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.core.errors import DomainError
from app.core.logging import get_logger

log = get_logger(__name__)

MAX_HISTORY = 20


class JobConflictError(DomainError):
    """Another admin job is already running."""

    status_code = 409
    code = "job_in_progress"


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    detail: str = ""
    processed: int = 0
    total: int = 0
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    async def report(self, processed: int, total: int, detail: str | None = None) -> None:
        self.processed = processed
        self.total = total
        if detail is not None:
            self.detail = detail


JobRunner = Callable[[Job], Awaitable[str]]


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._active: Job | None = None

    @property
    def active(self) -> Job | None:
        return self._active

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def recent(self, limit: int = 10) -> list[Job]:
        return list(reversed(list(self._jobs.values())))[:limit]

    def start(self, kind: str, runner: JobRunner) -> Job:
        if self._active is not None and self._active.status in ("queued", "running"):
            raise JobConflictError(
                f"job {self._active.id} ({self._active.kind}) is still running; "
                f"wait for it to finish"
            )

        job = Job(id=uuid.uuid4().hex[:12], kind=kind, status="running", detail="starting")
        self._jobs[job.id] = job
        while len(self._jobs) > MAX_HISTORY:
            self._jobs.popitem(last=False)
        self._active = job

        task = asyncio.create_task(self._execute(job, runner))
        # Hold a reference so the task is not garbage collected mid-flight.
        job_tasks.add(task)
        task.add_done_callback(job_tasks.discard)
        return job

    async def _execute(self, job: Job, runner: JobRunner) -> None:
        try:
            job.detail = await runner(job)
            job.status = "succeeded"
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            log.exception("job.failed", job_id=job.id, kind=job.kind)
        finally:
            job.finished_at = datetime.now(UTC)
            if self._active is job:
                self._active = None
            log.info("job.finished", job_id=job.id, kind=job.kind, status=job.status)


job_tasks: set[asyncio.Task[None]] = set()
_registry = JobRegistry()


def get_job_registry() -> JobRegistry:
    return _registry
