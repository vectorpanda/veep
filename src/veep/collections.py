"""Collection management for the veep SDK."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from .exceptions import CollectionAlreadyExistsError, ValidationError
from .models import Collection

if TYPE_CHECKING:
    from .client import VP

logger = logging.getLogger("veep")

VALID_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


class Collections:
    """Manage vector collections.

    Access this through ``client.collections`` -- do not instantiate directly.

    Example::

        vp = VP(api_key="...")
        vp.collections.create("products")
        cols = vp.collections.list()
        info = vp.collections.get("products")
        vp.collections.delete("products")
    """

    def __init__(self, client: VP):
        self._client = client

    def create(
        self,
        name: str,
        *,
        tier: str = "hot",
        id_field: str | None = None,
        vector_field: str | None = None,
        format: str | None = None,
        dimension: int | None = None,
    ) -> Collection:
        """Create a new collection.

        If you provide id_field and vector_field, the schema is locked in
        immediately and uploaded files will be processed without a manual
        confirmation step. If you omit them, Vector Panda will auto-detect
        the schema from your first upload and ask you to confirm.

        Args:
            name: Collection name. Letters, numbers, underscores, and hyphens only.
            tier: Storage tier -- 'hot' (default), 'warm', or 'paused'.
            id_field: Column name to use as the vector key/identifier.
            vector_field: Column name containing embedding vectors.
            format: Data format hint ('parquet', 'csv', 'jsonl'). Optional.
            dimension: Expected vector dimension. Optional.

        Returns:
            The newly created Collection.

        Raises:
            ValidationError: If the name contains invalid characters,
                or if only one of id_field/vector_field is provided.
            CollectionAlreadyExistsError: If a collection with this name already exists.
            AuthError: If your API key is invalid.
        """
        if not VALID_NAME.match(name):
            raise ValidationError(
                f"Collection name '{name}' is invalid. "
                f"Use only letters, numbers, underscores, and hyphens."
            )
        if tier not in ("hot", "warm", "paused"):
            raise ValidationError(
                f"Tier '{tier}' is not valid. Choose 'hot', 'warm', or 'paused'."
            )
        if bool(id_field) != bool(vector_field):
            raise ValidationError(
                "Provide both id_field and vector_field together, or neither."
            )

        body: dict = {"collection": name, "tier": tier}
        if id_field and vector_field:
            body["id_field"] = id_field
            body["vector_field"] = vector_field
            if format is not None:
                body["format"] = format
            if dimension is not None:
                body["dimension"] = dimension
            logger.info(
                "Creating collection '%s' (tier: %s, schema: %s/%s)...",
                name, tier, id_field, vector_field,
            )
        else:
            logger.info("Creating collection '%s' (tier: %s)...", name, tier)

        try:
            resp = self._client._request(
                "POST",
                "/api/v1/collections",
                json=body,
                accept_statuses=(201,),
            )
            data = resp.json()
            logger.info("Collection '%s' created.", name)
            return Collection(
                name=data.get("collection", name),
                tier=data.get("tier", tier),
                is_active=True,
                status="processing",
                dimension=dimension,
            )
        except CollectionAlreadyExistsError:
            logger.info("Collection '%s' already exists, returning existing.", name)
            return self.get(name)

    def get(self, name: str) -> Collection:
        """Get detailed information about a collection.

        Args:
            name: Collection name.

        Returns:
            A Collection with full details including vector count, dimension, and status.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            AuthError: If your API key is invalid.
        """
        logger.info("Getting details for collection '%s'...", name)
        resp = self._client._request("GET", f"/api/v1/collections/{name}")
        data = resp.json()
        return _parse_collection(data, fallback_name=name)

    def list(self) -> list[Collection]:
        """List all collections accessible to your API key.

        Returns:
            A list of Collection objects.

        Raises:
            AuthError: If your API key is invalid.
        """
        logger.info("Listing collections...")
        resp = self._client._request("GET", "/api/v1/collections")
        data = resp.json()
        items = data.get("collections", data if isinstance(data, list) else [])
        result = [_parse_collection(c) for c in items]
        logger.info("Found %d collection(s).", len(result))
        return result

    def delete(self, name: str) -> None:
        """Delete a collection and all its data.

        This permanently removes the collection, its files, and all indexed vectors.
        This action cannot be undone.

        Args:
            name: Collection name.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            AuthError: If your API key is invalid.
        """
        logger.info("Deleting collection '%s'...", name)
        self._client._request("DELETE", f"/api/v1/collections/{name}")
        logger.info("Collection '%s' deleted.", name)

    def status(self, name: str) -> str:
        """Check the processing status of a collection.

        This is a lightweight poll -- use it to check if uploads have been
        processed and the collection is ready for queries.

        Args:
            name: Collection name.

        Returns:
            Status string: 'unknown', 'processing', 'ready', or 'error'.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            AuthError: If your API key is invalid.
        """
        logger.info("Checking status of collection '%s'...", name)
        resp = self._client._request("GET", f"/api/v1/collections/{name}/status")
        data = resp.json()
        status = data.get("status", "unknown")
        logger.info("Collection '%s' status: %s", name, status)
        return status


def _parse_collection(data: dict, fallback_name: str = "") -> Collection:
    # server-0k60: vector_count / storage_gb / dimension may be null when
    # the collection is partially-created; pass None through.
    # server-ja0: failure_reason carries coord-side last_failure (e.g.
    # 'capacity_limited: ...') so callers can show stuck collections.
    return Collection(
        name=data.get("name") or fallback_name,
        tier=data.get("tier", "unknown"),
        is_active=data.get("is_active", True),
        vector_count=data.get("vector_count"),
        storage_gb=data.get("storage_gb"),
        status=data.get("status", "unknown"),
        dimension=data.get("dimension"),
        failure_reason=data.get("failure_reason"),
    )
