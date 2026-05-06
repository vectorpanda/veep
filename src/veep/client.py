"""VP client -- the main entry point for the veep SDK."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from .collections import Collections
from .exceptions import (
    AuthError,
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    FileAlreadyExistsError,
    NotFoundError,
    QueryError,
    ServerError,
    TimeoutError,
    ValidationError,
    VeepError,
)
from .schema import Schema
from .vectors import Vectors

DEFAULT_HOST = "https://api.vectorpanda.com"
DEFAULT_TIMEOUT = 120.0
# server-uls9: file uploads are inherently unbounded-size and routinely
# exceed the short-op timeout on multi-GB datasets. None means "no client-
# side deadline; let the request run to completion." Customers who want a
# hard upload cap can pass `upload_timeout=N` to VP(...).
DEFAULT_UPLOAD_TIMEOUT: float | None = None

# Sentinel for _request's timeout override — distinguishes "caller didn't
# specify" from "caller explicitly passed None (= no timeout)".
_UNSET = object()

logger = logging.getLogger("veep")


class VP:
    """Client for the Vector Panda vector search API.

    This is the only class you need to import. It gives you access to
    collections, vectors, and schema management through sub-resources.

    Three ways to initialize::

        # 1. With an API key (explicit or from environment)
        vp = VP(api_key="sk_live_...")

        # 2. Interactive login (opens browser for OAuth)
        vp = VP.login()

        # 3. From previously saved credentials
        vp = VP.from_creds()

    Args:
        api_key: Your API key. Falls back to the VEEP_API_KEY environment variable.
        host: API base URL. Falls back to VEEP_HOST, then to the Vector Panda cloud.
        timeout: Request timeout in seconds for short operations (queries,
            schema, listing, etc). Default is 120. Does NOT apply to file
            uploads — see ``upload_timeout``.
        upload_timeout: Request timeout in seconds for file uploads
            (``vectors.upsert(file=...)`` and ``vectors.replace(file=...)``).
            Default is ``None`` meaning no client-side deadline — the upload
            runs to completion regardless of size. Customers who want a hard
            cap (e.g. CI environments) can pass ``upload_timeout=1800``.
            (server-uls9: prior to 0.x.y, ``timeout`` applied to uploads too,
            which silently capped any single upload at roughly
            ``timeout × upload_speed`` — typically ~1 GB on a 10 MB/s link.
            Splitting the two timeouts removes that footgun without hiding
            connection hangs in the common short-op path.)
        verbose: If True, logs what the client is doing in plain English.

    Raises:
        AuthError: If no API key is provided or found in the environment.
    """

    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        upload_timeout: float | None = DEFAULT_UPLOAD_TIMEOUT,
        verbose: bool = False,
    ):
        self.api_key = api_key or os.environ.get("VEEP_API_KEY", "")
        if not self.api_key:
            raise AuthError(
                "API key required. Pass api_key= or set the VEEP_API_KEY environment variable.\n"
                "Or use VP.login() for interactive sign-in."
            )

        self.host = (host or os.environ.get("VEEP_HOST", DEFAULT_HOST)).rstrip("/")
        self.timeout = timeout
        self.upload_timeout = upload_timeout
        self.verbose = verbose

        if verbose and not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("veep: %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        elif not verbose:
            logger.setLevel(logging.WARNING)

        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {self.api_key}"

        self.collections = Collections(self)
        self.vectors = Vectors(self)
        self.schema = Schema(self)

        logger.info("Connected to %s", self.host)

    def ping(self) -> bool:
        """Check if the Vector Panda service is reachable.

        Returns:
            True if the service responds, False otherwise.
        """
        logger.info("Pinging service...")
        try:
            resp = self._session.get(
                f"{self.host}/api/v1/health",
                timeout=min(self.timeout, 5.0),
            )
            ok = resp.status_code == 200
            logger.info("Service is %s", "up" if ok else "unreachable")
            return ok
        except requests.RequestException:
            logger.info("Service is unreachable")
            return False

    @classmethod
    def login(
        cls,
        host: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        upload_timeout: float | None = DEFAULT_UPLOAD_TIMEOUT,
        verbose: bool = False,
        open_browser: bool = True,
        save: bool = True,
        force: bool = False,
    ) -> VP:
        """Sign in interactively and return a ready-to-use client.

        If ~/.veep/credentials.json exists and the saved key is still valid,
        login() returns immediately without opening a browser. Pass
        ``force=True`` to skip the saved-credential check and always run the
        device flow (e.g., to switch accounts).

        Opens a browser for OAuth (Google or GitHub). Works in terminals,
        Jupyter notebooks, and IPython. No need to copy-paste API keys.

        Args:
            host: API base URL. Defaults to VEEP_HOST or Vector Panda cloud.
                If saved credentials target a different host, the device flow
                runs against this host instead of reusing the saved creds.
            timeout: Request timeout for the returned client.
            verbose: Enable verbose logging on the returned client.
            open_browser: Automatically open the sign-in URL. Default True.
            save: Save credentials to ~/.veep/credentials.json. Default True.
            force: Skip the saved-credentials shortcut. Default False.

        Returns:
            An authenticated VP client.

        Example::

            vp = VP.login()
            print(vp.collections.list())
        """
        from .auth import clear_credentials, device_login, load_credentials, save_credentials

        # server-r255: short-circuit to saved credentials when they exist
        # and still authenticate. Falls through to the device flow only on
        # missing creds, host mismatch, or a 401/403 from the validation
        # probe. Transient network errors do NOT clear saved creds.
        if not force:
            saved = load_credentials()
            if saved and (host is None or saved.get("host") == host):
                client = cls(
                    api_key=saved["api_key"],
                    host=saved.get("host"),
                    timeout=timeout,
                    upload_timeout=upload_timeout,
                    verbose=verbose,
                )
                status = client._validate_credentials()
                if status == "ok":
                    logger.info("Reusing saved credentials from ~/.veep/credentials.json")
                    return client
                if status == "unauthorized":
                    logger.info("Saved credentials rejected (401/403); clearing and re-authenticating")
                    clear_credentials()

        result = device_login(host=host, open_browser=open_browser)

        if save:
            save_credentials(
                api_key=result["api_key"],
                host=result["host"],
                client_id=result.get("client_id", ""),
            )

        return cls(
            api_key=result["api_key"],
            host=result["host"],
            timeout=timeout,
            upload_timeout=upload_timeout,
            verbose=verbose,
        )

    def _validate_credentials(self) -> str:
        """Probe the API with the current key to check whether it still works.

        Returns one of:
          - "ok"            — server accepted the credentials (2xx).
          - "unauthorized"  — server rejected them (401/403).
          - "unreachable"   — network/timeout error, status unknown.
          - "unknown"       — server returned some other status; treat as ok-ish.
        """
        try:
            resp = self._session.get(
                f"{self.host}/api/v1/collections",
                timeout=min(self.timeout, 10.0),
                params={"limit": 1},
            )
        except requests.RequestException:
            return "unreachable"
        if resp.status_code in (401, 403):
            return "unauthorized"
        if 200 <= resp.status_code < 300:
            return "ok"
        return "unknown"

    @classmethod
    def from_creds(
        cls,
        timeout: float = DEFAULT_TIMEOUT,
        upload_timeout: float | None = DEFAULT_UPLOAD_TIMEOUT,
        verbose: bool = False,
    ) -> VP:
        """Load a client from previously saved credentials.

        Reads ~/.veep/credentials.json, written by login() or vp.save().

        Args:
            timeout: Request timeout in seconds.
            verbose: Enable verbose logging.

        Returns:
            An authenticated VP client.

        Raises:
            AuthError: If no saved credentials are found.
        """
        from .auth import load_credentials

        creds = load_credentials()
        if not creds:
            raise AuthError(
                "No saved credentials found. Run VP.login() first, "
                "or pass api_key= directly."
            )

        return cls(
            api_key=creds["api_key"],
            host=creds.get("host"),
            timeout=timeout,
            upload_timeout=upload_timeout,
            verbose=verbose,
        )

    def save(self) -> None:
        """Save this client's API key and host to ~/.veep/credentials.json.

        Future sessions can use VP.from_creds() to reconnect
        without passing the API key again.
        """
        from .auth import save_credentials
        save_credentials(api_key=self.api_key, host=self.host)

    def _reset_session(self) -> None:
        """Drop the current requests.Session and start fresh.

        server-t4d9: after an upstream 5xx, the urllib3 connection pool can
        retain a half-broken keepalive socket; subsequent requests then fail
        instantly with ConnectionError even though the upstream is healthy.
        Replacing the Session forces a clean reconnect.
        """
        try:
            self._session.close()
        except Exception:  # noqa: BLE001
            pass
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {self.api_key}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        data: Any = None,
        files: dict | None = None,
        params: dict | None = None,
        headers: dict | None = None,
        accept_statuses: tuple[int, ...] = (200,),
        timeout: Any = _UNSET,
        retries: int = 0,
    ) -> requests.Response:
        """Make an authenticated HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE).
            path: URL path (e.g., '/api/v1/collections').
            json: JSON body to send.
            data: Raw body data.
            files: Files to upload (multipart).
            params: Query parameters.
            headers: Extra headers to merge with the session's defaults
                (used by the chunked upload path to pass X-Content-Sha256).
            accept_statuses: HTTP status codes that count as success.

        Returns:
            The requests.Response object.

        Raises:
            AuthError: On 401 responses.
            ValidationError: On 400 responses.
            CollectionNotFoundError: On 404 for collection paths.
            CollectionAlreadyExistsError: On 409 for collection creation.
            FileAlreadyExistsError: On 409 for file uploads.
            NotFoundError: On 404 for other paths.
            TimeoutError: On request timeout.
            ServerError: On 5xx responses.
        """
        url = f"{self.host}{path}"
        # server-uls9: per-call timeout override, sentinel-defaulted so we
        # can tell "caller didn't specify" from "caller explicitly said None".
        effective_timeout = self.timeout if timeout is _UNSET else timeout

        # server-t4d9 + server-4i6b: bounded retry loop for transient
        # failures. Replaces the connection pool on ConnectionError /
        # ChunkedEncodingError (5xx aftershock that wedges the keepalive
        # socket); sleeps + retries on 502/503/504 without resetting.
        # Timeouts are NOT auto-retried — a POST that timed out may have
        # committed server-side, and a blind retry could double-write.
        # Callers that want retries opt in via `retries=N`.
        max_attempts = max(1, retries + 1)
        resp: requests.Response | None = None
        for attempt in range(max_attempts):
            try:
                resp = self._session.request(
                    method,
                    url,
                    json=json,
                    data=data,
                    files=files,
                    params=params,
                    headers=headers,
                    timeout=effective_timeout,
                )
            except requests.exceptions.Timeout:
                raise TimeoutError(
                    f"Request to {path} timed out after {effective_timeout} seconds. "
                    f"For short ops (query/schema/list): VP(timeout=N). "
                    f"For uploads: VP(upload_timeout=N) — default is no limit."
                ) from None
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError):
                if attempt + 1 < max_attempts:
                    self._reset_session()
                    time.sleep(min(0.5 * (2 ** attempt), 4.0))
                    continue
                raise ServerError(
                    f"Could not connect to {self.host}. "
                    f"Check that the host is correct and the service is running."
                ) from None

            if resp.status_code in (502, 503, 504) and attempt + 1 < max_attempts:
                time.sleep(min(0.5 * (2 ** attempt), 4.0))
                continue
            break

        assert resp is not None  # guaranteed: every loop iteration either sets resp or raises

        if resp.status_code in accept_statuses:
            return resp

        error_msg = _extract_error(resp)

        if resp.status_code == 401:
            raise AuthError(
                "Your API key was rejected. "
                "Check that it's correct and hasn't been rotated."
            )

        if resp.status_code == 400:
            raise ValidationError(error_msg)

        if resp.status_code == 404:
            collection = _extract_collection_from_path(path)
            if collection:
                raise CollectionNotFoundError(collection)
            raise NotFoundError(error_msg)

        if resp.status_code == 409:
            if "/files/" in path:
                parts = path.split("/")
                col_idx = parts.index("collections") + 1 if "collections" in parts else -1
                file_idx = parts.index("files") + 1 if "files" in parts else -1
                col = parts[col_idx] if col_idx > 0 and col_idx < len(parts) else "unknown"
                fname = parts[file_idx] if file_idx > 0 and file_idx < len(parts) else "unknown"
                raise FileAlreadyExistsError(col, fname)
            collection = _extract_collection_from_path(path)
            if collection:
                raise CollectionAlreadyExistsError(collection)
            if "collections" in path and json and "collection" in json:
                raise CollectionAlreadyExistsError(json["collection"])
            raise VeepError(error_msg)

        if resp.status_code == 502:
            raise QueryError(
                "The query service is temporarily unavailable. Try again in a moment."
            )

        if resp.status_code == 504:
            raise TimeoutError(
                "Your query took too long and was cancelled. "
                "Try reducing top_k or querying a smaller collection."
            )

        raise ServerError(error_msg, status_code=resp.status_code)


def _extract_error(resp: requests.Response) -> str:
    try:
        body = resp.json()
        return body.get("error", resp.text)
    except Exception:
        return resp.text or f"HTTP {resp.status_code}"


def _extract_collection_from_path(path: str) -> str | None:
    parts = path.split("/")
    if "collections" in parts:
        idx = parts.index("collections") + 1
        if idx < len(parts):
            return parts[idx]
    return None
