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

    This usually surfaces in the seconds right after an upload, while Vector Panda
    finishes indexing. The right fix is to wait a moment and try again — most
    collections become queryable within a few seconds of the upload finishing.

    Note: ``vp.vectors.upsert()`` blocks until the collection is queryable, so
    customers calling upsert and then immediately querying should never see this.
    You'll typically see it only when querying a collection that another process
    just modified, or after a paused collection is being resumed.
    """

    def __init__(self, name: str, status: str = "preparing", suggested_wait_seconds: float = 2.0):
        self.collection_name = name
        self.status = status
        self.suggested_wait_seconds = suggested_wait_seconds
        super().__init__(
            f"Collection '{name}' is still being prepared (status: {status}). "
            f"Vector Panda is finishing the index — usually a couple of seconds "
            f"after upload. Try the same call again in {suggested_wait_seconds:.0f} seconds."
        )


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
