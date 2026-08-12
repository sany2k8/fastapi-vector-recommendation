"""Domain exceptions. Services raise these; the app edge converts them to HTTP."""


class DomainError(Exception):
    """Base class for errors that map to a client-visible HTTP response."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(DomainError):
    status_code = 404
    code = "not_found"


class ValidationError(DomainError):
    status_code = 422
    code = "validation_error"


class EmbeddingError(DomainError):
    """The embedding provider failed or returned something unusable."""

    status_code = 502
    code = "embedding_failed"


class UnsupportedOperationError(DomainError):
    """The active configuration cannot serve this request (e.g. cross-modal on the stub)."""

    status_code = 409
    code = "unsupported_operation"
