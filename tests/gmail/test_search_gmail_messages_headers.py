"""Tests for search_gmail_messages header enrichment (include_headers flag)."""

from unittest.mock import Mock

import pytest

from gmail.gmail_tools import search_gmail_messages


def _unwrap(tool):
    """Unwrap FunctionTool + decorators to the original async function."""
    fn = tool.fn if hasattr(tool, "fn") else tool
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _headers(**overrides):
    header_map = {
        "Subject": "Example subject",
        "From": "sender@example.com",
        "Date": "Fri, 28 Mar 2026 10:00:00 -0400",
    }
    header_map.update(overrides)
    return [{"name": name, "value": value} for name, value in header_map.items()]


def _metadata_response(message_id: str, headers=None):
    return {
        "id": message_id,
        "payload": {"headers": headers or _headers()},
    }


class _FakeBatch:
    def __init__(self, callback):
        self._callback = callback
        self._requests = []

    def add(self, request, request_id):
        self._requests.append((request_id, request))

    def execute(self):
        for request_id, request in self._requests:
            try:
                response = request.execute()
                self._callback(request_id, response, None)
            except Exception as exc:
                self._callback(request_id, None, exc)


def _build_service(*, list_response, message_responses=None, batch_factory=None):
    message_responses = message_responses or {}

    service = Mock()

    def message_list(**kwargs):
        request = Mock()
        request.execute.return_value = list_response
        return request

    def message_get(**kwargs):
        request = Mock()
        response = message_responses[(kwargs["id"], kwargs["format"])]
        if isinstance(response, Exception):
            request.execute.side_effect = response
        else:
            request.execute.return_value = response
        return request

    service.users().messages().list.side_effect = message_list
    service.users().messages().get.side_effect = message_get
    if batch_factory is None:
        batch_factory = lambda callback: _FakeBatch(callback)
    service.new_batch_http_request.side_effect = batch_factory
    return service


def _list_response(message_ids, next_page_token=None):
    response = {
        "messages": [
            {"id": mid, "threadId": f"thread-{mid}"} for mid in message_ids
        ]
    }
    if next_page_token:
        response["nextPageToken"] = next_page_token
    return response


async def _run_search(service, **kwargs):
    return await _unwrap(search_gmail_messages)(
        service=service,
        query="test query",
        user_google_email="user@example.com",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_default_output_is_unchanged():
    """include_headers=False (the default) must produce today's exact format."""
    service = _build_service(list_response=_list_response(["msg-1", "msg-2"]))

    result = await _run_search(service)

    expected = "\n".join(
        [
            "Found 2 messages matching 'test query':",
            "",
            "📧 MESSAGES:",
            "  1. Message ID: msg-1",
            "     Web Link: https://mail.google.com/mail/u/0/#all/msg-1",
            "     Thread ID: thread-msg-1",
            "     Thread Link: https://mail.google.com/mail/u/0/#all/thread-msg-1",
            "",
            "  2. Message ID: msg-2",
            "     Web Link: https://mail.google.com/mail/u/0/#all/msg-2",
            "     Thread ID: thread-msg-2",
            "     Thread Link: https://mail.google.com/mail/u/0/#all/thread-msg-2",
            "",
            "💡 USAGE:",
            "  • Pass the Message IDs **as a list** to get_gmail_messages_content_batch()",
            "    e.g. get_gmail_messages_content_batch(message_ids=[...])",
            "  • Pass the Thread IDs to get_gmail_thread_content() (single) or get_gmail_threads_content_batch() (batch)",
        ]
    )
    assert result == expected


@pytest.mark.asyncio
async def test_default_makes_no_metadata_fetches():
    service = _build_service(list_response=_list_response(["msg-1"]))

    await _run_search(service)

    assert service.users.return_value.messages.return_value.get.call_count == 0
    assert service.new_batch_http_request.call_count == 0


@pytest.mark.asyncio
async def test_include_headers_returns_subject_sender_and_date():
    service = _build_service(
        list_response=_list_response(["msg-1", "msg-2"]),
        message_responses={
            ("msg-1", "metadata"): _metadata_response(
                "msg-1", headers=_headers(Subject="First subject")
            ),
            ("msg-2", "metadata"): _metadata_response(
                "msg-2",
                headers=_headers(
                    Subject="Second subject",
                    From="other@example.com",
                    Date="Sat, 29 Mar 2026 09:00:00 -0400",
                ),
            ),
        },
    )

    result = await _run_search(service, include_headers=True)

    assert "Subject: First subject" in result
    assert "Subject: Second subject" in result
    assert "From: sender@example.com" in result
    assert "From: other@example.com" in result
    assert "Date: Fri, 28 Mar 2026 10:00:00 -0400" in result
    assert "Date: Sat, 29 Mar 2026 09:00:00 -0400" in result
    # IDs and links are still present alongside the headers.
    assert "Message ID: msg-1" in result
    assert "Thread ID: thread-msg-1" in result

    formats = [
        call.kwargs["format"]
        for call in service.users.return_value.messages.return_value.get.call_args_list
    ]
    assert formats == ["metadata", "metadata"]


@pytest.mark.asyncio
async def test_partial_metadata_failure_degrades_per_row(monkeypatch):
    import gmail.gmail_tools as gmail_tools

    monkeypatch.setattr(gmail_tools, "GMAIL_REQUEST_DELAY", 0)
    service = _build_service(
        list_response=_list_response(["msg-1", "msg-2"]),
        message_responses={
            ("msg-1", "metadata"): _metadata_response("msg-1"),
            ("msg-2", "metadata"): Exception("boom"),
        },
    )

    result = await _run_search(service, include_headers=True)

    assert "Subject: Example subject" in result
    assert "Headers: unavailable (metadata fetch failed)" in result
    # The failing message's row survives with its IDs intact.
    assert "Message ID: msg-2" in result
    assert "Thread ID: thread-msg-2" in result


@pytest.mark.asyncio
async def test_transient_batch_failure_recovers_on_retry(monkeypatch):
    """A 429 inside the batch is retried sequentially and recovers."""
    import gmail.gmail_tools as gmail_tools

    monkeypatch.setattr(gmail_tools, "GMAIL_REQUEST_DELAY", 0)

    calls = {"count": 0}

    def flaky_get(**kwargs):
        request = Mock()
        calls["count"] += 1
        if calls["count"] == 1:
            request.execute.side_effect = Exception(
                "429 Too many concurrent requests for user."
            )
        else:
            request.execute.return_value = _metadata_response("msg-1")
        return request

    service = _build_service(list_response=_list_response(["msg-1"]))
    service.users().messages().get.side_effect = flaky_get

    result = await _run_search(service, include_headers=True)

    assert "Subject: Example subject" in result
    assert "Headers: unavailable" not in result
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_missing_headers_use_placeholders():
    service = _build_service(
        list_response=_list_response(["msg-1"]),
        message_responses={
            ("msg-1", "metadata"): {"id": "msg-1", "payload": {"headers": []}},
        },
    )

    result = await _run_search(service, include_headers=True)

    assert "Subject: (no subject)" in result
    assert "From: (unknown sender)" in result
    assert "Date: (unknown date)" in result


@pytest.mark.asyncio
async def test_pagination_preserved_with_include_headers():
    service = _build_service(
        list_response=_list_response(["msg-1"], next_page_token="tok-123"),
        message_responses={
            ("msg-1", "metadata"): _metadata_response("msg-1"),
        },
    )

    result = await _run_search(service, include_headers=True, page_token="tok-000")

    assert "page_token='tok-123'" in result
    assert "Subject: Example subject" in result
    list_kwargs = (
        service.users.return_value.messages.return_value.list.call_args.kwargs
    )
    assert list_kwargs["pageToken"] == "tok-000"


@pytest.mark.asyncio
async def test_batch_failure_falls_back_to_sequential(monkeypatch):
    import gmail.gmail_tools as gmail_tools

    monkeypatch.setattr(gmail_tools, "GMAIL_REQUEST_DELAY", 0)

    def _broken_batch(callback):
        raise RuntimeError("batch endpoint unavailable")

    service = _build_service(
        list_response=_list_response(["msg-1"]),
        message_responses={
            ("msg-1", "metadata"): _metadata_response("msg-1"),
        },
        batch_factory=_broken_batch,
    )

    result = await _run_search(service, include_headers=True)

    assert "Subject: Example subject" in result
    assert "From: sender@example.com" in result
