"""Collection management for the veep SDK."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .exceptions import (
    CollectionAlreadyExistsError,
    CollectionRecentlyDeletedError,
    ServerError,
    TimeoutError,
    ValidationError,
)
from .models import Collection, ExportResult

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
        if_exists: str = "ignore",
        target_recall: float | None = None,
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
            target_recall: Recall target for this collection, between 0.50
                and 0.999 (default 0.95). Vector Panda serves the fastest
                search configuration whose measured recall meets this
                target -- raise it for maximum accuracy, lower it for
                maximum speed. Change it later with ``update()``.
            if_exists: What to do when a collection with this name already
                exists OR was recently deleted (the name is in a brief
                post-delete cooldown). One of:

                - ``"ignore"`` (default): if the collection exists, log a
                  warning and return it. The ``tier`` and schema args you
                  passed are NOT applied to the existing collection. If
                  the name is in post-delete cooldown, raises
                  :class:`CollectionRecentlyDeletedError` (there's no
                  existing collection to return).
                - ``"replace"``: DESTRUCTIVE. Bypass both gates — replaces
                  any existing collection (and ALL its stored vectors)
                  AND clears the post-delete cooldown. Use only when you
                  intentionally want a clean slate.
                - ``"error"``: raise on either gate
                  (:class:`CollectionAlreadyExistsError` or
                  :class:`CollectionRecentlyDeletedError`).
                  Strict opt-in for callers that need fail-fast behavior.

        Returns:
            The newly created Collection (or the existing one when
            ``if_exists="ignore"`` and the collection already existed).

        Raises:
            ValidationError: If the name contains invalid characters, if only
                one of id_field/vector_field is provided, or if ``if_exists``
                is not one of the three allowed values.
            CollectionAlreadyExistsError: When ``if_exists="error"`` and a
                collection with this name already exists.
            CollectionRecentlyDeletedError: When the name is in the brief
                post-delete cooldown and ``if_exists`` is not ``"replace"``.
                Wait the suggested seconds and retry, or pass
                ``if_exists="replace"`` to bypass.
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
        if if_exists not in ("ignore", "replace", "error"):
            raise ValidationError(
                f"if_exists must be 'ignore', 'replace', or 'error'; got '{if_exists}'."
            )
        if target_recall is not None and not (0.50 <= target_recall <= 0.999):
            raise ValidationError(
                f"target_recall must be between 0.50 and 0.999; got {target_recall}."
            )

        # server-or56: if_exists="replace" maps to force_destroy=true on the
        # wire. Coord's create gate refuses to bypass an existing row OR an
        # in-cooldown tombstone unless force is set; the SDK's "replace"
        # contract is the customer's explicit "yes, take the bigger hammer."
        force = if_exists == "replace"

        body: dict = {"collection": name, "tier": tier}
        if force:
            body["force_destroy"] = True
        if target_recall is not None:
            body["target_recall"] = target_recall
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

        def _do_create() -> Collection:
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

        try:
            return _do_create()
        except CollectionAlreadyExistsError:
            if if_exists == "error":
                raise
            if if_exists == "replace":
                # Should not happen — replace passes force_destroy=true and
                # coord UPSERTs in place. If we got here, surface honestly.
                raise
            # if_exists == "ignore" (default)
            logger.warning(
                "Collection '%s' already exists. Returning the existing "
                "collection. The 'tier' and schema arguments you passed were "
                "NOT applied. To replace it (DELETES ALL STORED VECTORS), call "
                "vp.collections.create('%s', if_exists='replace'). To delete "
                "first, call vp.collections.delete('%s'). To raise an "
                "exception on conflict instead, pass if_exists='error'.",
                name, name, name,
            )
            return self.get(name)
        except CollectionRecentlyDeletedError:
            # Post-delete cooldown — there's no existing collection to
            # return for "ignore" mode, so it surfaces the same way as
            # "error" mode. "replace" sent force_destroy=true and should
            # never trip this; if it does, surface honestly.
            raise

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

    def export(
        self,
        name: str,
        path: str | Path,
        *,
        wait: bool = True,
        send_email: bool | str = False,
        input_dim: int | None = None,
        poll_interval_s: float = 2.0,
        timeout_s: float | None = None,
    ) -> ExportResult:
        """Export a collection to a directory of parquet parts.

        Your data is yours. This writes every vector's original id,
        Float32 vector at the original dim, and metadata into a
        directory you can read with::

            duckdb.read_parquet('out/*.parquet')
            pd.read_parquet('out/')

        Snapshot-at-start: the export reflects the collection at the
        moment the job started; writes that arrive during the export
        will be in the next one.

        Args:
            name: Collection name.
            path: Destination directory. Created if missing.
            wait: Block until every part is on disk (default). ``False``
                returns immediately after the job is queued; the
                returned :class:`ExportResult` carries only ``job_id``.
            send_email: Email on completion. ``False`` (default) sends
                no email; ``True`` sends to the account email on file;
                a string sends to that explicit address.
            input_dim: Optional hint of the original vector dim. The
                server derives it from on-disk artifacts when omitted.
            poll_interval_s: Seconds between status polls when
                ``wait=True``. Default 2s.
            timeout_s: Total client-side wait deadline. ``None``
                (default) means wait indefinitely — exports of large
                collections can take a while and the server-side cap
                only triggers on a stalled job.

        Returns:
            An :class:`ExportResult`. When ``wait=True``, ``path``,
            ``parts``, ``total_bytes`` are populated. When ``wait=False``,
            only ``job_id`` is set.

        Raises:
            ValidationError: Invalid name, send_email shape, or
                interval/dim argument.
            CollectionNotFoundError: Collection does not exist.
            ServerError: Export job failed server-side.
            TimeoutError: ``timeout_s`` elapsed before the job completed.
            AuthError: API key invalid.
        """
        if not VALID_NAME.match(name):
            raise ValidationError(
                f"Collection name '{name}' is invalid. "
                f"Use only letters, numbers, underscores, and hyphens."
            )
        if isinstance(send_email, bool):
            pass
        elif isinstance(send_email, str):
            if not send_email.strip():
                raise ValidationError(
                    "send_email must be False, True, or a non-empty email address string."
                )
        else:
            raise ValidationError(
                "send_email must be False (no email), True (account email), "
                "or an email address string."
            )
        if input_dim is not None and (not isinstance(input_dim, int) or input_dim <= 0):
            raise ValidationError("input_dim must be a positive integer.")
        if poll_interval_s <= 0:
            raise ValidationError("poll_interval_s must be greater than 0.")
        if timeout_s is not None and timeout_s <= 0:
            raise ValidationError("timeout_s must be greater than 0, or None to wait indefinitely.")

        body: dict[str, Any] = {}
        if send_email is True:
            body["send_email"] = True
        elif isinstance(send_email, str):
            body["send_email"] = send_email.strip()
        if input_dim is not None:
            body["input_dim"] = input_dim

        logger.info("Starting export of collection '%s'...", name)
        resp = self._client._request(
            "POST",
            f"/api/v1/collections/{name}/export-jobs",
            json=body,
            accept_statuses=(202,),
        )
        job_id = resp.json()["job_id"]
        logger.info("Export job '%s' queued.", job_id)

        if not wait:
            return ExportResult(job_id=job_id, path=None, parts=0, total_bytes=0, status="rolling")

        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)

        deadline = (time.time() + timeout_s) if timeout_s is not None else None
        while True:
            row = self._get_export_job(name, job_id)
            status = row.get("status", "unknown")
            if status == "complete":
                break
            if status == "failed":
                err = row.get("error") or "no error message"
                raise ServerError(f"Export job '{job_id}' failed: {err}")
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(
                    f"Export job '{job_id}' did not complete within {timeout_s:.0f}s "
                    f"(last status: '{status}'). The job continues running on the server — "
                    f"re-run with wait=False to track it manually, or pass a larger timeout_s."
                )
            time.sleep(poll_interval_s)

        manifest_resp = self._client._request(
            "GET",
            f"/api/v1/collections/{name}/export-jobs/{job_id}/manifest",
        )
        manifest = manifest_resp.json()
        parts = manifest.get("parts", []) or []

        # server-z78b: prefix sidecars with "_" so parquet readers
        # (pyarrow/pandas dataset discovery) ignore them by convention and
        # `pd.read_parquet(dest)` / `pq.read_table(dest)` treat the directory
        # as one logical table — pointing any reader at the export dir "just
        # works" without globbing around manifest.json / the README.
        (dest / "_manifest.json").write_text(json.dumps(manifest, indent=2))

        total_bytes = 0
        for idx, part in enumerate(parts):
            filename = part.get("filename") or f"part-{idx:04d}.parquet"
            target = dest / filename
            logger.info("Downloading %s (%d bytes)...", filename, part.get("bytes", 0))
            url = f"{self._client.host}/api/v1/collections/{name}/export-jobs/{job_id}/download"
            with self._client._session.get(
                url,
                params={"part": idx},
                stream=True,
                timeout=self._client.upload_timeout,
            ) as r:
                if r.status_code != 200:
                    raise ServerError(
                        f"Failed to download part {idx} ({filename}): HTTP {r.status_code}"
                    )
                with target.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            total_bytes += target.stat().st_size

        # EXPORT_README.md sidecar — best-effort. Older servers without
        # the readme writer will 503; that's fine, the customer still
        # has the parts + manifest.
        try:
            readme_url = (
                f"{self._client.host}/api/v1/collections/{name}"
                f"/export-jobs/{job_id}/download"
            )
            with self._client._session.get(
                readme_url,
                params={"file": "EXPORT_README.md"},
                timeout=self._client.upload_timeout,
            ) as r:
                if r.status_code == 200:
                    # server-z78b: "_" prefix so parquet readers skip it (see _manifest.json above).
                    (dest / "_EXPORT_README.md").write_bytes(r.content)
                elif r.status_code == 503:
                    logger.debug("EXPORT_README.md not present on this export")
                else:
                    logger.warning(
                        "EXPORT_README.md fetch returned HTTP %s", r.status_code,
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning("EXPORT_README.md fetch failed: %s", e)

        logger.info(
            "Export complete: %d part(s), %d bytes written to %s",
            len(parts), total_bytes, dest,
        )
        return ExportResult(
            job_id=job_id,
            path=dest,
            parts=len(parts),
            total_bytes=total_bytes,
            status="complete",
        )

    def _get_export_job(self, collection: str, job_id: str) -> dict:
        resp = self._client._request(
            "GET",
            f"/api/v1/collections/{collection}/export-jobs/{job_id}",
        )
        return resp.json()

    def update(self, name: str, *, target_recall: float) -> Collection:
        """Update a collection's settings.

        Args:
            name: Collection name.
            target_recall: New recall target, between 0.50 and 0.999.
                Vector Panda serves the fastest search configuration whose
                measured recall meets this target. Raising it trades speed
                for accuracy; lowering it trades accuracy for speed. The
                change takes effect immediately and the serving
                configuration is re-evaluated under the new target.

        Returns:
            The updated Collection.

        Raises:
            ValidationError: If target_recall is outside 0.50-0.999.
            CollectionNotFoundError: If the collection does not exist.
            AuthError: If your API key is invalid.
        """
        if not (0.50 <= target_recall <= 0.999):
            raise ValidationError(
                f"target_recall must be between 0.50 and 0.999; got {target_recall}."
            )
        logger.info(
            "Updating collection '%s' (target_recall: %s)...", name, target_recall
        )
        self._client._request(
            "PATCH",
            f"/api/v1/collections/{name}",
            json={"target_recall": target_recall},
        )
        logger.info("Collection '%s' updated.", name)
        return self.get(name)

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
        # server-rg2e: auto-optimizer convergence state; poll for
        # 'index_optimized' to know the index is finalized.
        optimization_state=data.get("optimization_state"),
        # server-obhw: the collection's recall target.
        target_recall=data.get("target_recall"),
    )
