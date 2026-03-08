"""HTTP retry logic for LLM provider requests.

Handles transient errors with exponential backoff:
- 429 Too Many Requests (rate limit)
- 500 Internal Server Error
- 502 Bad Gateway
- 503 Service Unavailable
- 504 Gateway Timeout
- Connection errors (server down, DNS failure)
- Timeout errors

Non-retryable errors are raised immediately:
- 400 Bad Request (malformed request — fix it, don't retry)
- 401 Unauthorized (bad API key)
- 403 Forbidden
- 404 Not Found
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger("nexagen.http")

# Status codes worth retrying
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Default retry config
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0       # seconds
DEFAULT_MAX_DELAY = 30.0       # seconds
DEFAULT_BACKOFF_FACTOR = 2.0


class RetryConfig:
    """Configuration for HTTP retry behavior."""

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    retry_config: RetryConfig | None = None,
    **kwargs,
) -> httpx.Response:
    """Make an HTTP request with retry logic for transient failures.

    Args:
        client: httpx.AsyncClient to use
        method: HTTP method ("POST", "GET", etc.)
        url: Request URL
        retry_config: Retry configuration (uses defaults if None)
        **kwargs: Additional kwargs passed to client.request()

    Returns:
        httpx.Response on success

    Raises:
        httpx.HTTPStatusError: On non-retryable HTTP errors (4xx except 429)
        httpx.ConnectError: If all retries exhausted on connection errors
        httpx.TimeoutException: If all retries exhausted on timeouts
    """
    config = retry_config or RetryConfig()
    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            response = await client.request(method, url, **kwargs)

            # Non-retryable client errors — raise immediately
            if 400 <= response.status_code < 500 and response.status_code != 429:
                response.raise_for_status()

            # Retryable status codes
            if response.status_code in _RETRYABLE_STATUS_CODES:
                if attempt < config.max_retries:
                    delay = _calculate_delay(attempt, response, config)
                    logger.warning(
                        "HTTP %d from %s (attempt %d/%d). Retrying in %.1fs...",
                        response.status_code, url, attempt + 1, config.max_retries + 1, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Final attempt — raise the error
                    response.raise_for_status()

            # Success
            return response

        except httpx.TimeoutException as e:
            last_exception = e
            if attempt < config.max_retries:
                delay = _calculate_delay(attempt, None, config)
                logger.warning(
                    "Timeout on %s (attempt %d/%d). Retrying in %.1fs...",
                    url, attempt + 1, config.max_retries + 1, delay,
                )
                await asyncio.sleep(delay)
                continue

        except httpx.ConnectError as e:
            last_exception = e
            if attempt < config.max_retries:
                delay = _calculate_delay(attempt, None, config)
                logger.warning(
                    "Connection error on %s (attempt %d/%d). Retrying in %.1fs...",
                    url, attempt + 1, config.max_retries + 1, delay,
                )
                await asyncio.sleep(delay)
                continue

    # All retries exhausted
    if last_exception:
        raise last_exception

    # Should never reach here, but just in case
    raise httpx.ConnectError(f"All {config.max_retries + 1} attempts failed for {url}")


def _calculate_delay(
    attempt: int,
    response: httpx.Response | None,
    config: RetryConfig,
) -> float:
    """Calculate retry delay with exponential backoff + Retry-After support."""
    # Check for Retry-After header (429 rate limits often include this)
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), config.max_delay)
            except ValueError:
                pass

    # Exponential backoff: base_delay * (backoff_factor ^ attempt)
    delay = config.base_delay * (config.backoff_factor ** attempt)
    return min(delay, config.max_delay)
