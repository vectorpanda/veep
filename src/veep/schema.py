"""Schema management for the veep SDK."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .exceptions import ValidationError
from .models import SchemaInfo

if TYPE_CHECKING:
    from .client import VP

logger = logging.getLogger("veep")


class Schema:
    """Inspect and confirm collection schemas.

    Access this through ``client.schema`` -- do not instantiate directly.

    After uploading files, Vector Panda analyzes them to detect the schema
    (which column has the vectors, which has the keys, etc.). You can inspect
    the detected schema and confirm it.

    Example::

        vp = VP(api_key="...")
        schema = vp.schema.get("my_collection")
        print(schema.state, schema.vector_field)
        vp.schema.confirm("my_collection", id_field="id", vector_field="embedding")
    """

    def __init__(self, client: VP):
        self._client = client

    def get(self, collection: str) -> SchemaInfo:
        """Get the current schema state for a collection.

        Args:
            collection: Collection name.

        Returns:
            A SchemaInfo with the detected or confirmed schema fields.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            AuthError: If your API key is invalid.
        """
        logger.info("Getting schema for collection '%s'...", collection)
        resp = self._client._request(
            "GET",
            f"/api/v1/collections/{collection}/schema",
        )
        data = resp.json()
        return SchemaInfo(
            state=data.get("state", "unknown"),
            id_field=data.get("id_field"),
            vector_field=data.get("vector_field"),
            format=data.get("format"),
            dimension=data.get("dimension"),
            analyzed=data.get("samples_analyzed", 0),
            pending=data.get("pending_count", 0),
        )

    def confirm(
        self,
        collection: str,
        *,
        id_field: str,
        vector_field: str,
        format: str | None = None,
        dimension: int | None = None,
    ) -> dict:
        """Confirm the schema for a collection.

        Call this after uploading files to tell Vector Panda which columns
        contain your vector keys and embeddings. If the auto-detected schema
        is correct, you can skip this step.

        Args:
            collection: Collection name.
            id_field: The column name to use as the vector key/identifier.
            vector_field: The column name containing embedding vectors.
            format: Data format hint (e.g., 'parquet'). Optional.
            dimension: Expected vector dimension. Optional.

        Returns:
            Server confirmation response.

        Raises:
            ValidationError: If id_field or vector_field is empty.
            CollectionNotFoundError: If the collection does not exist.
            AuthError: If your API key is invalid.
        """
        if not id_field:
            raise ValidationError("id_field is required.")
        if not vector_field:
            raise ValidationError("vector_field is required.")

        logger.info(
            "Confirming schema for '%s' (id=%s, vector=%s)...",
            collection,
            id_field,
            vector_field,
        )

        body: dict = {"idField": id_field, "vectorField": vector_field}
        if format is not None:
            body["format"] = format
        if dimension is not None:
            body["dimension"] = dimension

        resp = self._client._request(
            "POST",
            f"/api/v1/collections/{collection}/schema/confirm",
            json=body,
        )
        logger.info("Schema confirmed for '%s'.", collection)
        return resp.json()

    def update(
        self,
        collection: str,
        *,
        id_field: str | None = None,
        vector_field: str | None = None,
        reprocess: bool = False,
    ) -> dict:
        """Update the schema field mappings for a collection.

        If the collection has existing data, you must pass ``reprocess=True``
        to rewrite source files with the new field names and rebuild artifacts.

        Args:
            collection: Collection name.
            id_field: New column name for the vector key. None to keep current.
            vector_field: New column name for embedding vectors. None to keep current.
            reprocess: If True, rewrite source files and rebuild artifacts.

        Returns:
            Server response with success status and details.

        Raises:
            ValidationError: If neither id_field nor vector_field is provided.
            VeepError: If reprocessing is required but reprocess=False (HTTP 409).
        """
        if id_field is None and vector_field is None:
            raise ValidationError("Provide at least one of id_field or vector_field.")

        logger.info(
            "Updating schema for '%s' (id=%s, vector=%s, reprocess=%s)...",
            collection,
            id_field,
            vector_field,
            reprocess,
        )

        body: dict = {"reprocess": reprocess}
        if id_field is not None:
            body["id_field"] = id_field
        if vector_field is not None:
            body["vector_field"] = vector_field

        resp = self._client._request(
            "POST",
            f"/api/v1/collections/{collection}/schema/update",
            json=body,
            accept_statuses=(200, 409),
        )

        data = resp.json()
        if resp.status_code == 409:
            logger.warning("Schema update requires reprocessing: %s", data.get("detail", ""))
        else:
            logger.info("Schema updated for '%s'.", collection)
            # server-5pbm.1.2.4: server emits a 'warning' field on soft-failure
            # success (e.g. no-op call where the requested fields match what's
            # already confirmed; reprocess=True that rewrote zero files because
            # the old field name didn't match any column). Surface to the
            # customer at WARNING level so they catch the mistake before
            # relying on the (technically-successful) update.
            if isinstance(data, dict):
                warning = data.get("warning")
                if warning:
                    logger.warning("%s", warning)
        return data
