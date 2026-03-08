"""Tests for HTTP retry logic."""

import asyncio

import httpx
import pytest

from nexagen.http_retry import (
    RetryConfig,
    request_with_retry,
    _calculate_delay,
    _RETRYABLE_STATUS_CODES,
)


class MockTransport(httpx.AsyncBaseTransport):
    """Mock transport that returns predefined responses in sequence."""

    def __init__(self, responses: list[httpx.Response]):
        self.responses = responses
        self.call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        idx = min(self.call_count, len(self.responses) - 1)
        response = self.responses[idx]
        response.request = request  # httpx requires this
        self.call_count += 1
        return response


class MockTransportWithError(httpx.AsyncBaseTransport):
    """Mock transport that raises errors then succeeds."""

    def __init__(self, errors: list[Exception], success_response: httpx.Response):
        self.errors = errors
        self.success_response = success_response
        self.call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.call_count < len(self.errors):
            self.call_count += 1
            raise self.errors[self.call_count - 1]
        self.call_count += 1
        self.success_response.request = request
        return self.success_response


def _make_response(status_code: int, json_data: dict | None = None, headers: dict | None = None) -> httpx.Response:
    """Create a mock httpx.Response."""
    resp = httpx.Response(
        status_code=status_code,
        json=json_data or {},
        headers=headers or {},
    )
    return resp


# ── RetryConfig ───────────────────────────────────────────────


class TestRetryConfig:
    def test_defaults(self):
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.backoff_factor == 2.0

    def test_custom(self):
        config = RetryConfig(max_retries=5, base_delay=0.5)
        assert config.max_retries == 5
        assert config.base_delay == 0.5


# ── Delay Calculation ─────────────────────────────────────────


class TestDelayCalculation:
    def test_exponential_backoff(self):
        config = RetryConfig(base_delay=1.0, backoff_factor=2.0)
        assert _calculate_delay(0, None, config) == 1.0    # 1 * 2^0
        assert _calculate_delay(1, None, config) == 2.0    # 1 * 2^1
        assert _calculate_delay(2, None, config) == 4.0    # 1 * 2^2
        assert _calculate_delay(3, None, config) == 8.0    # 1 * 2^3

    def test_max_delay_cap(self):
        config = RetryConfig(base_delay=1.0, backoff_factor=2.0, max_delay=5.0)
        assert _calculate_delay(10, None, config) == 5.0   # capped

    def test_retry_after_header(self):
        config = RetryConfig()
        response = _make_response(429, headers={"Retry-After": "3"})
        assert _calculate_delay(0, response, config) == 3.0

    def test_retry_after_capped(self):
        config = RetryConfig(max_delay=10.0)
        response = _make_response(429, headers={"Retry-After": "60"})
        assert _calculate_delay(0, response, config) == 10.0


# ── Successful Requests ──────────────────────────────────────


class TestSuccessfulRequests:
    async def test_success_no_retry(self):
        transport = MockTransport([_make_response(200, {"result": "ok"})])
        async with httpx.AsyncClient(transport=transport) as client:
            response = await request_with_retry(client, "POST", "http://test.com/api")
        assert response.status_code == 200
        assert transport.call_count == 1

    async def test_success_after_retry(self):
        """500 on first attempt, 200 on second."""
        transport = MockTransport([
            _make_response(500),
            _make_response(200, {"result": "ok"}),
        ])
        config = RetryConfig(base_delay=0.01)  # fast for tests
        async with httpx.AsyncClient(transport=transport) as client:
            response = await request_with_retry(client, "POST", "http://test.com/api", retry_config=config)
        assert response.status_code == 200
        assert transport.call_count == 2


# ── Retryable Status Codes ───────────────────────────────────


class TestRetryableStatusCodes:
    async def test_429_retried(self):
        transport = MockTransport([
            _make_response(429),
            _make_response(200),
        ])
        config = RetryConfig(base_delay=0.01)
        async with httpx.AsyncClient(transport=transport) as client:
            response = await request_with_retry(client, "POST", "http://test.com/api", retry_config=config)
        assert response.status_code == 200
        assert transport.call_count == 2

    async def test_502_retried(self):
        transport = MockTransport([
            _make_response(502),
            _make_response(200),
        ])
        config = RetryConfig(base_delay=0.01)
        async with httpx.AsyncClient(transport=transport) as client:
            response = await request_with_retry(client, "POST", "http://test.com/api", retry_config=config)
        assert response.status_code == 200

    async def test_503_retried(self):
        transport = MockTransport([
            _make_response(503),
            _make_response(503),
            _make_response(200),
        ])
        config = RetryConfig(base_delay=0.01)
        async with httpx.AsyncClient(transport=transport) as client:
            response = await request_with_retry(client, "POST", "http://test.com/api", retry_config=config)
        assert response.status_code == 200
        assert transport.call_count == 3

    async def test_all_retries_exhausted_raises(self):
        transport = MockTransport([
            _make_response(500),
            _make_response(500),
            _make_response(500),
            _make_response(500),  # 4th = beyond max_retries=3
        ])
        config = RetryConfig(max_retries=3, base_delay=0.01)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await request_with_retry(client, "POST", "http://test.com/api", retry_config=config)
            assert exc_info.value.response.status_code == 500
        assert transport.call_count == 4  # initial + 3 retries


# ── Non-Retryable Status Codes ────────────────────────────────


class TestNonRetryableStatusCodes:
    async def test_400_not_retried(self):
        transport = MockTransport([_make_response(400)])
        config = RetryConfig(base_delay=0.01)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await request_with_retry(client, "POST", "http://test.com/api", retry_config=config)
            assert exc_info.value.response.status_code == 400
        assert transport.call_count == 1  # no retries

    async def test_401_not_retried(self):
        transport = MockTransport([_make_response(401)])
        config = RetryConfig(base_delay=0.01)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await request_with_retry(client, "POST", "http://test.com/api", retry_config=config)
        assert transport.call_count == 1

    async def test_403_not_retried(self):
        transport = MockTransport([_make_response(403)])
        config = RetryConfig(base_delay=0.01)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await request_with_retry(client, "POST", "http://test.com/api", retry_config=config)
        assert transport.call_count == 1

    async def test_404_not_retried(self):
        transport = MockTransport([_make_response(404)])
        config = RetryConfig(base_delay=0.01)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await request_with_retry(client, "POST", "http://test.com/api", retry_config=config)
        assert transport.call_count == 1


# ── Connection Errors ─────────────────────────────────────────


class TestConnectionErrors:
    async def test_connection_error_retried(self):
        transport = MockTransportWithError(
            errors=[httpx.ConnectError("Connection refused")],
            success_response=_make_response(200),
        )
        config = RetryConfig(base_delay=0.01)
        async with httpx.AsyncClient(transport=transport) as client:
            response = await request_with_retry(client, "POST", "http://test.com/api", retry_config=config)
        assert response.status_code == 200
        assert transport.call_count == 2

    async def test_connection_error_exhausted(self):
        transport = MockTransportWithError(
            errors=[
                httpx.ConnectError("fail 1"),
                httpx.ConnectError("fail 2"),
                httpx.ConnectError("fail 3"),
                httpx.ConnectError("fail 4"),
            ],
            success_response=_make_response(200),
        )
        config = RetryConfig(max_retries=3, base_delay=0.01)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.ConnectError):
                await request_with_retry(client, "POST", "http://test.com/api", retry_config=config)


# ── Timeout Errors ────────────────────────────────────────────


class TestTimeoutErrors:
    async def test_timeout_retried(self):
        transport = MockTransportWithError(
            errors=[httpx.ReadTimeout("timed out")],
            success_response=_make_response(200),
        )
        config = RetryConfig(base_delay=0.01)
        async with httpx.AsyncClient(transport=transport) as client:
            response = await request_with_retry(client, "POST", "http://test.com/api", retry_config=config)
        assert response.status_code == 200

    async def test_timeout_exhausted(self):
        transport = MockTransportWithError(
            errors=[
                httpx.ReadTimeout("t1"),
                httpx.ReadTimeout("t2"),
                httpx.ReadTimeout("t3"),
                httpx.ReadTimeout("t4"),
            ],
            success_response=_make_response(200),
        )
        config = RetryConfig(max_retries=3, base_delay=0.01)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.TimeoutException):
                await request_with_retry(client, "POST", "http://test.com/api", retry_config=config)


# ── Retryable Status Code Set ─────────────────────────────────


class TestRetryableSet:
    def test_retryable_codes(self):
        assert 429 in _RETRYABLE_STATUS_CODES
        assert 500 in _RETRYABLE_STATUS_CODES
        assert 502 in _RETRYABLE_STATUS_CODES
        assert 503 in _RETRYABLE_STATUS_CODES
        assert 504 in _RETRYABLE_STATUS_CODES

    def test_non_retryable_codes(self):
        assert 400 not in _RETRYABLE_STATUS_CODES
        assert 401 not in _RETRYABLE_STATUS_CODES
        assert 403 not in _RETRYABLE_STATUS_CODES
        assert 404 not in _RETRYABLE_STATUS_CODES
        assert 200 not in _RETRYABLE_STATUS_CODES
