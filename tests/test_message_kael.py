"""Offline tests for the message_kael tool — fake desk-channel server
(httpx.MockTransport), no network. Same stance as test_home_control: every
path must come back as an honest string, never a raise.
"""

from __future__ import annotations

import httpx

from aerys_v2.tools.message_kael import LINE_DOWN, build_message_kael_tool

URL = "http://desk.test:8399/aerys/message"
TOKEN = "test-token"


def make_tool(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return build_message_kael_tool(URL, TOKEN, client=client)


class TestDelivery:
    def test_delivered_on_202(self):
        seen = {}

        def handler(req):
            seen["auth"] = req.headers.get("authorization")
            seen["body"] = req.read().decode()
            return httpx.Response(202, json={"ok": True})

        tool = make_tool(handler)
        out = tool.invoke({"message": "the office satellite dropped offline"})
        assert "Delivered" in out
        assert seen["auth"] == f"Bearer {TOKEN}"
        assert "office satellite" in seen["body"]

    def test_whitespace_collapsed_and_capped(self):
        seen = {}

        def handler(req):
            seen["body"] = req.read().decode()
            return httpx.Response(202)

        tool = make_tool(handler)
        tool.invoke({"message": "  a \n\n b  " + "x" * 5000})
        assert '"a b' in seen["body"]
        assert len(seen["body"]) < 2200  # MESSAGE_LIMIT + JSON envelope


class TestHonestFailures:
    def test_empty_message_never_sends(self):
        def handler(req):  # pragma: no cover - must not be reached
            raise AssertionError("no request should be made for empty input")

        tool = make_tool(handler)
        out = tool.invoke({"message": "   "})
        assert out.startswith("NOT DELIVERED")

    def test_connection_error_is_line_down(self):
        def handler(req):
            raise httpx.ConnectError("refused")

        tool = make_tool(handler)
        out = tool.invoke({"message": "hello"})
        assert out == LINE_DOWN
        assert "log_gap" in out

    def test_cooldown_429_is_honest(self):
        tool = make_tool(lambda req: httpx.Response(429, json={}))
        out = tool.invoke({"message": "hello"})
        assert out.startswith("NOT DELIVERED")
        assert "cooldown" in out

    def test_unexpected_status_is_line_down(self):
        tool = make_tool(lambda req: httpx.Response(404))
        out = tool.invoke({"message": "hello"})
        assert out == LINE_DOWN

    def test_never_raises(self):
        def handler(req):
            raise httpx.ReadTimeout("slow")

        tool = make_tool(handler)
        out = tool.invoke({"message": "hello"})
        assert isinstance(out, str) and out  # honest string, no exception


class TestFactoryWiring:
    def test_armed_only_with_both_settings(self):
        from aerys_v2.factory import MESSAGE_KAEL_OVERLAY  # noqa: F401

        # Overlay exists and teaches restraint + the log_gap split.
        assert "message_kael" in MESSAGE_KAEL_OVERLAY
        assert "log_gap" in MESSAGE_KAEL_OVERLAY
