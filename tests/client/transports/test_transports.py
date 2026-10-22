import contextlib
import ssl
from collections.abc import AsyncIterator
from ssl import VerifyMode
from typing import Any, cast
from unittest.mock import patch

import httpx2
import pytest
from mcp.shared._httpx_utils import McpHttpClientFactory

from fastmcp import Client
from fastmcp.client.auth.oauth import OAuth
from fastmcp.client.transports import (
    ClientTransport,
    SSETransport,
    StreamableHttpTransport,
)
from fastmcp.client.transports.base import TransportOptions


class BasicTransport(ClientTransport):
    @contextlib.asynccontextmanager
    async def connect_session(self, **session_kwargs: Any) -> AsyncIterator[Any]:
        raise AssertionError("BasicTransport does not create sessions")
        yield


class TestClientTransport:
    def test_default_repr_uses_subclass_name(self):
        assert repr(BasicTransport()) == "<BasicTransport>"

    def test_default_session_id_is_none(self):
        assert BasicTransport().get_session_id() is None

    def test_client_rejects_auth_for_transports_without_auth_support(self):
        with pytest.raises(ValueError, match="does not support auth"):
            Client(BasicTransport(), auth="oauth")

    def test_client_accepts_none_auth_for_transports_without_auth_support(self):
        client = Client(BasicTransport(), auth=None)

        assert isinstance(client.transport, BasicTransport)


async def test_oauth_uses_same_client_as_transport_streamable_http():
    transport = StreamableHttpTransport(
        "https://some.fake.url/",
        httpx_client_factory=lambda *args, **kwargs: httpx2.AsyncClient(
            verify=False, *args, **kwargs
        ),
        auth="oauth",
    )

    assert isinstance(transport.auth, OAuth)
    async with transport.auth.httpx_client_factory() as httpx_client:
        assert httpx_client._transport is not None
        assert (
            httpx_client._transport._pool._ssl_context.verify_mode  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
            == VerifyMode.CERT_NONE
        )


async def test_oauth_uses_same_client_as_transport_sse():
    transport = SSETransport(
        "https://some.fake.url/",
        httpx_client_factory=lambda *args, **kwargs: httpx2.AsyncClient(
            verify=False, *args, **kwargs
        ),
        auth="oauth",
    )

    assert isinstance(transport.auth, OAuth)
    async with transport.auth.httpx_client_factory() as httpx_client:
        assert httpx_client._transport is not None
        assert (
            httpx_client._transport._pool._ssl_context.verify_mode  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
            == VerifyMode.CERT_NONE
        )


class TestSSLVerify:
    def test_streamable_http_transport_stores_verify_false(self):
        transport = StreamableHttpTransport(
            "https://example.com/mcp",
            verify=False,
        )
        assert transport.verify is False

    def test_streamable_http_transport_stores_verify_ssl_context(self):
        ctx = ssl.create_default_context()
        transport = StreamableHttpTransport(
            "https://example.com/mcp",
            verify=ctx,
        )
        assert transport.verify is ctx

    def test_streamable_http_transport_stores_verify_cert_path(self):
        transport = StreamableHttpTransport(
            "https://example.com/mcp",
            verify="/path/to/cert.pem",
        )
        assert transport.verify == "/path/to/cert.pem"

    def test_streamable_http_transport_verify_default_is_none(self):
        transport = StreamableHttpTransport("https://example.com/mcp")
        assert transport.verify is None

    def test_sse_transport_stores_verify_false(self):
        transport = SSETransport(
            "https://example.com/sse",
            verify=False,
        )
        assert transport.verify is False

    def test_sse_transport_stores_verify_ssl_context(self):
        ctx = ssl.create_default_context()
        transport = SSETransport(
            "https://example.com/sse",
            verify=ctx,
        )
        assert transport.verify is ctx

    def test_sse_transport_verify_default_is_none(self):
        transport = SSETransport("https://example.com/sse")
        assert transport.verify is None

    def test_client_passes_verify_to_streamable_http_transport(self):
        client = Client("https://example.com/mcp", verify=False)
        assert isinstance(client.transport, StreamableHttpTransport)
        assert client.transport.verify is False

    def test_client_passes_verify_ssl_context_to_transport(self):
        ctx = ssl.create_default_context()
        client = Client("https://example.com/mcp", verify=ctx)
        assert isinstance(client.transport, StreamableHttpTransport)
        assert client.transport.verify is ctx

    def test_client_passes_verify_cert_path_to_transport(self):
        client = Client(
            "https://example.com/mcp",
            verify="/path/to/cert.pem",
        )
        assert isinstance(client.transport, StreamableHttpTransport)
        assert client.transport.verify == "/path/to/cert.pem"

    def test_client_verify_none_leaves_transport_default(self):
        client = Client("https://example.com/mcp")
        assert isinstance(client.transport, StreamableHttpTransport)
        assert client.transport.verify is None

    def test_client_verify_raises_for_non_http_transport(self):
        from fastmcp import FastMCP

        server = FastMCP("test")
        with pytest.raises(
            ValueError,
            match="only supported for HTTP transports",
        ):
            Client(server, verify=False)

    def test_client_passes_verify_to_sse_transport(self):
        client = Client("https://example.com/sse", verify=False)
        assert isinstance(client.transport, SSETransport)
        assert client.transport.verify is False

    async def test_streamable_http_verify_propagates_to_oauth(self):
        transport = StreamableHttpTransport(
            "https://example.com/mcp",
            verify=False,
            auth="oauth",
        )
        assert isinstance(transport.auth, OAuth)
        async with transport.auth.httpx_client_factory() as httpx_client:
            assert (
                httpx_client._transport._pool._ssl_context.verify_mode  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
                == VerifyMode.CERT_NONE
            )

    async def test_sse_verify_propagates_to_oauth(self):
        transport = SSETransport(
            "https://example.com/sse",
            verify=False,
            auth="oauth",
        )
        assert isinstance(transport.auth, OAuth)
        async with transport.auth.httpx_client_factory() as httpx_client:
            assert (
                httpx_client._transport._pool._ssl_context.verify_mode  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
                == VerifyMode.CERT_NONE
            )

    async def test_client_verify_propagates_to_oauth(self):
        client = Client(
            "https://example.com/mcp",
            verify=False,
            auth="oauth",
        )
        assert isinstance(client.transport, StreamableHttpTransport)
        assert isinstance(client.transport.auth, OAuth)
        async with client.transport.auth.httpx_client_factory() as httpx_client:
            assert (
                httpx_client._transport._pool._ssl_context.verify_mode  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
                == VerifyMode.CERT_NONE
            )

    async def test_verify_propagates_to_preconstructed_oauth_instance(self):
        transport = StreamableHttpTransport(
            "https://example.com/mcp",
            verify=False,
            auth=OAuth(),
        )
        assert isinstance(transport.auth, OAuth)
        async with transport.auth.httpx_client_factory() as httpx_client:
            assert (
                httpx_client._transport._pool._ssl_context.verify_mode  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
                == VerifyMode.CERT_NONE
            )

    async def test_client_verify_resyncs_existing_oauth_on_transport(self):
        transport = StreamableHttpTransport(
            "https://example.com/mcp",
            auth="oauth",
        )
        assert isinstance(transport.auth, OAuth)
        # OAuth was created without verify — factory should be default
        async with transport.auth.httpx_client_factory() as httpx_client:
            assert (
                httpx_client._transport._pool._ssl_context.verify_mode  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
                != VerifyMode.CERT_NONE
            )

        # Now wrap in Client with verify=False — should resync OAuth
        client = Client(transport, verify=False)
        assert isinstance(client.transport.auth, OAuth)
        async with client.transport.auth.httpx_client_factory() as httpx_client:
            assert (
                httpx_client._transport._pool._ssl_context.verify_mode  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
                == VerifyMode.CERT_NONE
            )

    async def test_client_verify_overrides_transport_verify_in_oauth(self):
        transport = StreamableHttpTransport(
            "https://example.com/mcp",
            verify=False,
            auth="oauth",
        )
        assert isinstance(transport.auth, OAuth)
        # OAuth should initially have verify=False
        async with transport.auth.httpx_client_factory() as httpx_client:
            assert (
                httpx_client._transport._pool._ssl_context.verify_mode  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
                == VerifyMode.CERT_NONE
            )

        # Client overrides verify to True — OAuth should update
        client = Client(transport, verify=True)
        assert isinstance(client.transport.auth, OAuth)
        async with client.transport.auth.httpx_client_factory() as httpx_client:
            assert (
                httpx_client._transport._pool._ssl_context.verify_mode  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
                != VerifyMode.CERT_NONE
            )

    async def test_oauth_custom_factory_preserved_with_verify(self):
        custom_factory = cast(
            McpHttpClientFactory,
            lambda **kwargs: httpx2.AsyncClient(verify=False, **kwargs),
        )
        auth = OAuth(httpx_client_factory=custom_factory)
        transport = StreamableHttpTransport(
            "https://example.com/mcp",
            verify=True,
            auth=auth,
        )
        assert isinstance(transport.auth, OAuth)
        assert transport.auth.httpx_client_factory is custom_factory

    def test_warns_when_both_factory_and_verify_provided_streamable(self):
        factory = cast(McpHttpClientFactory, httpx2.AsyncClient)
        with pytest.warns(UserWarning, match="httpx_client_factory.*takes precedence"):
            StreamableHttpTransport(
                "https://example.com/mcp",
                httpx_client_factory=factory,
                verify=False,
            )

    def test_warns_when_both_factory_and_verify_provided_sse(self):
        factory = cast(McpHttpClientFactory, httpx2.AsyncClient)
        with pytest.warns(UserWarning, match="httpx_client_factory.*takes precedence"):
            SSETransport(
                "https://example.com/sse",
                httpx_client_factory=factory,
                verify=False,
            )


class TestForwardedHeaderCasePrecedence:
    """Configured transport headers must win over forwarded headers regardless of case.

    get_http_headers returns lowercase keys (e.g. "authorization") while the
    transport may be configured with mixed-case headers (e.g. "Authorization").
    Before the fix, both ended up in the merged dict because the dict merge
    compared keys literally, so the backend received two header lines for the
    same logical header. The configured header must always take precedence.
    """

    def _captured_headers_factory(self, captured: dict) -> McpHttpClientFactory:
        """Return an httpx_client_factory that records the headers it received."""

        def factory(
            headers: dict[str, str] | None = None,
            auth: httpx2.Auth | None = None,
            follow_redirects: bool = True,
            **kwargs: Any,
        ) -> httpx2.AsyncClient:
            captured["headers"] = dict(headers or {})
            return httpx2.AsyncClient(headers=headers or {})

        return cast(McpHttpClientFactory, factory)

    async def test_streamable_http_configured_header_wins_over_forwarded_lowercase(
        self,
    ):
        """When Authorization (mixed case) is configured, the forwarded
        authorization (lowercase) must not appear alongside it."""
        captured: dict = {}
        transport = StreamableHttpTransport(
            "https://example.com/mcp",
            headers={"Authorization": "Bearer backend"},
            httpx_client_factory=self._captured_headers_factory(captured),
        )
        options = TransportOptions(forward_incoming_headers=True)

        with patch(
            "fastmcp.client.transports.http.get_http_headers",
            return_value={"authorization": "Bearer caller"},
        ):
            try:
                async with transport.connect_session(transport_options=options):
                    pass
            except Exception:
                pass  # connection will fail; we only care about the headers built

        assert captured.get("headers") is not None
        headers = captured["headers"]
        # The forwarded lowercase key must not appear alongside the configured key
        assert "authorization" not in headers or headers.get("authorization") == headers.get(
            "Authorization"
        ), "Both authorization and Authorization were present with different values"
        # The configured value must be present (under whichever key the transport uses)
        header_values = {v for v in headers.values()}
        assert "Bearer backend" in header_values
        assert "Bearer caller" not in header_values

    async def test_streamable_http_configured_header_wins_exact_case_match(self):
        """When both sides use lowercase, the configured header still wins."""
        captured: dict = {}
        transport = StreamableHttpTransport(
            "https://example.com/mcp",
            headers={"authorization": "Bearer backend"},
            httpx_client_factory=self._captured_headers_factory(captured),
        )
        options = TransportOptions(forward_incoming_headers=True)

        with patch(
            "fastmcp.client.transports.http.get_http_headers",
            return_value={"authorization": "Bearer caller"},
        ):
            try:
                async with transport.connect_session(transport_options=options):
                    pass
            except Exception:
                pass

        assert captured.get("headers") is not None
        headers = captured["headers"]
        header_values = list(headers.values())
        assert "Bearer backend" in header_values
        assert "Bearer caller" not in header_values

    async def test_streamable_http_forwarded_header_included_when_not_overridden(self):
        """A forwarded header that is not overridden by a configured header is kept."""
        captured: dict = {}
        transport = StreamableHttpTransport(
            "https://example.com/mcp",
            headers={"X-Backend-Token": "backend-only"},
            httpx_client_factory=self._captured_headers_factory(captured),
        )
        options = TransportOptions(forward_incoming_headers=True)

        with patch(
            "fastmcp.client.transports.http.get_http_headers",
            return_value={"authorization": "Bearer caller"},
        ):
            try:
                async with transport.connect_session(transport_options=options):
                    pass
            except Exception:
                pass

        assert captured.get("headers") is not None
        headers = captured["headers"]
        header_values = list(headers.values())
        assert "Bearer caller" in header_values
        assert "backend-only" in header_values
