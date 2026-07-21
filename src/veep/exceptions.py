"""Exception hierarchy for veep SDK.

Every exception tells you what went wrong, what you probably did wrong,
and what to do about it.
"""

from __future__ import annotations


class VeepError(Exception):
    """Base exception for all veep errors."""


class AuthError(VeepError):
    """Your API key is invalid, missing, or expired.

    Fix: check that you passed the correct api_key to VP(),
    or that VEEP_API_KEY is set in your environment.
    """


class ValidationError(VeepError):
    """Something about your request was invalid.

    The message tells you exactly which parameter was wrong and why.
    """


class NotFoundError(VeepError):
    """The resource you asked for does not exist.

    This usually means a typo in a collection name or filename.
    """


class CollectionNotFoundError(NotFoundError):
    """The collection you asked for does not exist.

    Check the name with client.collections.list() to see what's available.
    """

    def __init__(self, name: str):
        self.collection_name = name
        super().__init__(
            f"Collection '{name}' not found. "
            f"Use client.collections.list() to see available collections."
        )


class CollectionNotReadyError(VeepError):
    """The collection exists but is still being prepared and can't serve queries yet.

    Typically raised from two places:

    - **Query / fetch on a not-yet-ready collection** — surfaces in the seconds
      right after an upload while Vector Panda finishes indexing. Wait a moment
      and retry.
    - **Upsert wait-timeout** — ``upsert()`` blocks until the collection
      transitions to ``ready``; if the ingest pipeline doesn't reach ``ready``
      within the configured wait, the SDK raises this error rather than letting
      your code hang forever. The right response is *not* to retry the upsert
      (that would create duplicate state); call ``vp.collections.status(name)``
      to see the current state, and contact support if it stays stuck.
    """

    def __init__(
        self,
        name: str,
        status: str = "preparing",
        suggested_wait_seconds: float = 2.0,
        message: str | None = None,
    ):
        self.collection_name = name
        self.status = status
        self.suggested_wait_seconds = suggested_wait_seconds
        if message is None:
            message = (
                f"Collection '{name}' is still being prepared (status: {status}). "
                f"Vector Panda is finishing the index — usually a couple of seconds "
                f"after upload. Try the same call again in {suggested_wait_seconds:.0f} seconds."
            )
        super().__init__(message)


class CollectionAlreadyExistsError(VeepError):
    """You tried to create a collection that already exists.

    If you want to use the existing collection, just skip the create step.
    If you want to replace it, delete it first with client.collections.delete().
    """

    def __init__(self, name: str):
        self.collection_name = name
        super().__init__(
            f"Collection '{name}' already exists. "
            f"Delete it first with client.collections.delete('{name}') if you want to recreate it."
        )


class CollectionRecentlyDeletedError(VeepError):
    """The collection name you tried to create is in a brief post-delete cooldown.

    Vector Panda holds the name for a short window after delete to filter
    out any in-flight signals from the prior collection — without this,
    a delete-then-create cycle can race and leave you with a partially-
    initialized new collection. Two ways forward:

    - Wait the suggested seconds and retry the create normally, OR
    - Pass ``if_exists="replace"`` (or ``force_destroy=True`` on the raw
      HTTP API) to bypass the cooldown immediately.
    """

    def __init__(self, name: str, retry_after_secs: int | None = None):
        self.collection_name = name
        self.retry_after_secs = retry_after_secs
        wait_hint = (
            f"retry in {retry_after_secs}s" if retry_after_secs else "retry in a moment"
        )
        super().__init__(
            f"Collection '{name}' was recently deleted; {wait_hint} or call "
            f"client.collections.create('{name}', if_exists='replace') to bypass the cooldown."
        )


class UploadError(VeepError):
    """Something went wrong uploading your file.

    The message tells you what happened. Common causes:
    - File does not exist on disk
    - File is not a supported format
    - File already exists (use replace instead of upsert)
    """


class FileAlreadyExistsError(UploadError):
    """This file already exists in the collection.

    Use client.vectors.replace() instead of client.vectors.upsert()
    to overwrite an existing file.
    """

    def __init__(self, collection: str, filename: str):
        self.collection = collection
        self.filename = filename
        super().__init__(
            f"File '{filename}' already exists in collection '{collection}'. "
            f"Use client.vectors.replace('{collection}', '{filename}') to overwrite it."
        )


class QueryError(VeepError):
    """Your query could not be completed.

    Common causes:
    - The collection is not ready yet (still processing uploads)
    - The vector dimension doesn't match the collection
    """


class TimeoutError(VeepError):
    """The request timed out.

    This can happen with large queries or when the service is under heavy load.
    Try again, or increase the timeout with VP(timeout=300).
    """


class ServerError(VeepError):
    """Something unexpected went wrong on the server side.

    This is not your fault. If it persists, contact support.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
