"""Device authorization flow and credential persistence for veep.

Enables fully programmatic login from Python/Jupyter without
manually copying API keys:

    from veep import VP
    vp = VP.login()  # opens browser, completes OAuth, returns client

Credentials are saved to ~/.veep/credentials.json and reused
on subsequent sessions via VP.from_creds().
"""

from __future__ import annotations

import json
import logging
import os
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable

import requests

logger = logging.getLogger("veep")

CREDENTIALS_DIR = Path.home() / ".veep"
CREDENTIALS_FILE = CREDENTIALS_DIR / "credentials.json"

DEFAULT_HOST = "https://api.vectorpanda.com"


def save_credentials(api_key: str, host: str | None = None, **extra: Any) -> Path:
    """Save API key and host to ~/.veep/credentials.json.

    Args:
        api_key: The API key to save.
        host: The API host URL. Defaults to the Vector Panda cloud.
        **extra: Additional fields to store (e.g., client_id).

    Returns:
        Path to the credentials file.
    """
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    data = {"api_key": api_key, "host": host or DEFAULT_HOST, **extra}
    CREDENTIALS_FILE.write_text(json.dumps(data, indent=2) + "\n")
    CREDENTIALS_FILE.chmod(0o600)
    logger.info("Credentials saved to %s", CREDENTIALS_FILE)
    return CREDENTIALS_FILE


def load_credentials() -> dict | None:
    """Load saved credentials from ~/.veep/credentials.json.

    Returns:
        Dict with at least 'api_key' and 'host', or None if not found.
    """
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        data = json.loads(CREDENTIALS_FILE.read_text())
        if "api_key" in data:
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def clear_credentials() -> None:
    """Remove saved credentials."""
    if CREDENTIALS_FILE.exists():
        CREDENTIALS_FILE.unlink()
        logger.info("Credentials cleared")


def _is_notebook() -> bool:
    """Detect if running in a Jupyter/IPython notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def _display_link(url: str, user_code: str) -> None:
    """Display the verification URL — clickable in notebooks, printed in terminals."""
    if _is_notebook():
        try:
            from IPython.display import HTML, display
            display(HTML(
                f'<p>Open this link to sign in: <a href="{url}" target="_blank">{url}</a></p>'
                f'<p>Your confirmation code: <b>{user_code}</b></p>'
            ))
            return
        except ImportError:
            pass
    print(f"\nOpen this URL to sign in:\n  {url}\n")
    print(f"Your confirmation code: {user_code}\n")


def device_login(
    host: str | None = None,
    open_browser: bool = True,
    timeout_s: int = 300,
    on_device_code: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Run the device authorization flow.

    1. Requests a device code from the server
    2. Opens the verification URL in a browser (or prints it)
    3. Polls until the user completes OAuth in the browser
    4. Returns {"api_key": ..., "client_id": ..., "host": ...}

    Args:
        host: API base URL. Defaults to VEEP_HOST env var or Vector Panda cloud.
        open_browser: Whether to automatically open the browser. Default True.
        timeout_s: How long to wait for the user to complete login. Default 300s.
        on_device_code: Optional callback fired with (device_code, verification_url)
            after the device endpoint returns and before polling begins. Useful for
            programmatic flows that want to handle the verification URL (e.g., emit
            it via a different transport, automate approval in tests).

    Returns:
        Dict with api_key, client_id, and host.

    Raises:
        TimeoutError: If the user doesn't complete login in time.
        ServerError: If the device flow is not supported or fails.
    """
    from .exceptions import ServerError, TimeoutError

    host = (host or os.environ.get("VEEP_HOST", DEFAULT_HOST)).rstrip("/")

    # Step 1: Request device code
    try:
        resp = requests.post(
            f"{host}/api/v1/auth/device",
            json={},
            timeout=10,
        )
    except requests.exceptions.ConnectionError:
        raise ServerError(
            f"Could not connect to {host}. "
            f"Check that the host is correct and the service is running."
        ) from None

    if resp.status_code != 200:
        raise ServerError(
            f"Device login not available at {host}. "
            f"You may need to update your server or use VP(api_key=...) instead.",
            status_code=resp.status_code,
        )

    data = resp.json()
    device_code = data["device_code"]
    user_code = data["user_code"]
    verification_url = data["verification_url"]
    interval = data.get("interval", 5)
    expires_in = data.get("expires_in", timeout_s)

    if on_device_code is not None:
        on_device_code(device_code, verification_url)

    # Step 2: Show the URL to the user (suppressed when on_device_code handles it)
    if on_device_code is None:
        _display_link(verification_url, user_code)

    # server-sqom.9: when on_device_code is set, the caller has opted into
    # programmatic handling of the verification URL — auto-opening a browser
    # in addition is a surprise side-effect (the URL gets dispatched twice,
    # via the callback AND a browser tab the caller didn't ask for). Suppress
    # the browser the same way we suppress stdout.
    if open_browser and on_device_code is None:
        try:
            webbrowser.open(verification_url)
        except Exception:
            print("Could not open browser automatically. Please open the URL above.")

    if on_device_code is None:
        print("Waiting for sign-in...", end="", flush=True)

    # Step 3: Poll for completion
    deadline = time.time() + min(expires_in, timeout_s)
    while time.time() < deadline:
        time.sleep(interval)
        if on_device_code is None:
            print(".", end="", flush=True)

        try:
            poll_resp = requests.post(
                f"{host}/api/v1/auth/device/token",
                json={"device_code": device_code},
                timeout=10,
            )
        except requests.exceptions.RequestException:
            continue

        if poll_resp.status_code == 200:
            result = poll_resp.json()
            if on_device_code is None:
                print(" done!\n")
            logger.info("Login successful. Client ID: %s", result.get("client_id"))
            return {
                "api_key": result["api_key"],
                "client_id": result.get("client_id", ""),
                "host": host,
            }

        if poll_resp.status_code == 400:
            body = poll_resp.json()
            error = body.get("error", "")
            if error == "authorization_pending":
                continue
            if error == "expired_token":
                break
            if error == "access_denied":
                if on_device_code is None:
                    print(" denied.\n")
                raise ServerError("Login was denied by the user.")

    if on_device_code is None:
        print(" timed out.\n")
    raise TimeoutError(
        f"Login was not completed within {timeout_s} seconds. "
        f"Run VP.login() again to retry."
    )
