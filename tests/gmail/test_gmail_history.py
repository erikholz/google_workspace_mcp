"""Tests for get_gmail_history, the incremental-sync tool.

The correctness property that matters most here is that an empty change set,
an expired checkpoint and a truncated walk are three visibly different
answers. Gmail returns HTTP 404 for a checkpoint older than its retention
window, and a tool that swallowed that would report "nothing changed" for a
mailbox that had changed a great deal.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from googleapiclient.errors import HttpError

from core.utils import UserInputError
from gmail.gmail_helpers import _collect_history_changes
from gmail.gmail_tools import get_gmail_history


def _unwrap(tool):
    """Unwrap FunctionTool + decorators to the original async function."""
    fn = tool.fn if hasattr(tool, "fn") else tool
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _http_error(status: int) -> HttpError:
    response = MagicMock()
    response.status = status
    return HttpError(resp=response, content=b"{}")


def _msg(message_id: str, thread_id: str) -> dict:
    return {"message": {"id": message_id, "threadId": thread_id}}


def _labelled(message_id: str, thread_id: str, label_ids: list[str]) -> dict:
    return {"message": {"id": message_id, "threadId": thread_id}, "labelIds": label_ids}


def _build_service(pages: list[dict], profile_history_id: str = "99999") -> MagicMock:
    """Service whose history().list() returns the given pages in order."""
    service = MagicMock()
    service.users.return_value.getProfile.return_value.execute.return_value = {
        "emailAddress": "erik@example.com",
        "historyId": profile_history_id,
    }
    execute = MagicMock(side_effect=list(pages))
    service.users.return_value.history.return_value.list.return_value.execute = execute
    return service


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------


class TestCollectHistoryChanges:
    def test_buckets_and_counts(self):
        records = [
            {"id": "1", "messagesAdded": [_msg("m1", "t1")]},
            {"id": "2", "messagesDeleted": [_msg("m2", "t2")]},
            {"id": "3", "labelsAdded": [_labelled("m3", "t3", ["Label_1"])]},
            {"id": "4", "labelsRemoved": [_labelled("m4", "t4", ["Label_2"])]},
        ]
        result = _collect_history_changes(records)
        assert result["counts"] == {
            "messages_added": 1,
            "messages_deleted": 1,
            "labels_added": 1,
            "labels_removed": 1,
            "threads": 4,
        }
        assert result["messages_added"] == [{"id": "m1", "thread_id": "t1"}]
        assert result["labels_added"][0]["label_ids"] == ["Label_1"]

    def test_same_message_across_records_is_deduplicated(self):
        records = [
            {"id": "1", "messagesAdded": [_msg("m1", "t1")]},
            {"id": "2", "messagesAdded": [_msg("m1", "t1")]},
        ]
        result = _collect_history_changes(records)
        assert result["counts"]["messages_added"] == 1
        assert result["counts"]["threads"] == 1

    def test_label_ids_accumulate_across_records_without_duplicates(self):
        records = [
            {"id": "1", "labelsAdded": [_labelled("m1", "t1", ["A"])]},
            {"id": "2", "labelsAdded": [_labelled("m1", "t1", ["B", "A"])]},
        ]
        result = _collect_history_changes(records)
        assert result["labels_added"] == [
            {"id": "m1", "thread_id": "t1", "label_ids": ["A", "B"]}
        ]

    def test_thread_ids_are_a_deduplicated_union_in_first_seen_order(self):
        records = [
            {"id": "1", "messagesAdded": [_msg("m1", "t9"), _msg("m2", "t1")]},
            {"id": "2", "labelsRemoved": [_labelled("m3", "t9", ["X"])]},
        ]
        result = _collect_history_changes(records)
        assert result["thread_ids"] == ["t9", "t1"]

    def test_entries_without_a_message_id_are_skipped(self):
        records = [{"id": "1", "messagesAdded": [{"message": {}}, _msg("m1", "t1")]}]
        result = _collect_history_changes(records)
        assert result["counts"]["messages_added"] == 1

    def test_empty_input(self):
        result = _collect_history_changes([])
        assert result["thread_ids"] == []
        assert result["counts"]["threads"] == 0


# ---------------------------------------------------------------------------
# Tool modes
# ---------------------------------------------------------------------------


class TestBootstrapMode:
    @pytest.mark.asyncio
    async def test_no_checkpoint_returns_current_history_id_and_no_changes(self):
        service = _build_service([], profile_history_id="12345")
        result = await _unwrap(get_gmail_history)(
            service=service, user_google_email="erik@example.com"
        )
        assert result["mode"] == "bootstrap"
        assert result["history_id"] == "12345"
        assert result["changes"] is None

    @pytest.mark.asyncio
    async def test_bootstrap_does_not_call_the_history_endpoint(self):
        service = _build_service([], profile_history_id="12345")
        await _unwrap(get_gmail_history)(
            service=service, user_google_email="erik@example.com"
        )
        service.users.return_value.history.return_value.list.assert_not_called()


class TestDeltaMode:
    @pytest.mark.asyncio
    async def test_single_page_returns_changes_and_a_new_checkpoint(self):
        pages = [
            {
                "history": [{"id": "101", "messagesAdded": [_msg("m1", "t1")]}],
                "historyId": "500",
            }
        ]
        service = _build_service(pages)
        result = await _unwrap(get_gmail_history)(
            service=service,
            user_google_email="erik@example.com",
            start_history_id="100",
        )
        assert result["mode"] == "delta"
        assert result["complete"] is True
        assert result["expired"] is False
        assert result["history_id"] == "500"
        assert result["start_history_id"] == "100"
        assert result["pages"] == 1
        assert result["thread_ids"] == ["t1"]
        assert result["changes"]["messages_added"] == [{"id": "m1", "thread_id": "t1"}]

    @pytest.mark.asyncio
    async def test_no_changes_is_a_complete_answer_not_a_failure(self):
        service = _build_service([{"historyId": "500"}])
        result = await _unwrap(get_gmail_history)(
            service=service,
            user_google_email="erik@example.com",
            start_history_id="500",
        )
        assert result["mode"] == "delta"
        assert result["complete"] is True
        assert result["expired"] is False
        assert result["history_id"] == "500"
        assert result["counts"]["threads"] == 0
        assert result["thread_ids"] == []

    @pytest.mark.asyncio
    async def test_walks_every_page(self):
        pages = [
            {
                "history": [{"id": "1", "messagesAdded": [_msg("m1", "t1")]}],
                "historyId": "500",
                "nextPageToken": "p2",
            },
            {
                "history": [{"id": "2", "messagesAdded": [_msg("m2", "t2")]}],
                "historyId": "500",
            },
        ]
        service = _build_service(pages)
        result = await _unwrap(get_gmail_history)(
            service=service,
            user_google_email="erik@example.com",
            start_history_id="100",
        )
        assert result["pages"] == 2
        assert result["complete"] is True
        assert result["thread_ids"] == ["t1", "t2"]

    @pytest.mark.asyncio
    async def test_label_and_type_filters_reach_the_api(self):
        service = _build_service([{"historyId": "500"}])
        await _unwrap(get_gmail_history)(
            service=service,
            user_google_email="erik@example.com",
            start_history_id="100",
            label_id="Label_42",
            history_types=["messageAdded"],
        )
        kwargs = service.users.return_value.history.return_value.list.call_args.kwargs
        assert kwargs["labelId"] == "Label_42"
        assert kwargs["historyTypes"] == ["messageAdded"]
        assert kwargs["startHistoryId"] == "100"


class TestExpiredCheckpoint:
    @pytest.mark.asyncio
    async def test_404_is_reported_as_expired_with_a_fresh_checkpoint(self):
        """The failure that matters: never report an expired checkpoint as
        'nothing changed'."""
        service = _build_service([], profile_history_id="77777")
        service.users.return_value.history.return_value.list.return_value.execute = (
            MagicMock(side_effect=_http_error(404))
        )

        result = await _unwrap(get_gmail_history)(
            service=service,
            user_google_email="erik@example.com",
            start_history_id="1",
        )
        assert result["mode"] == "expired"
        assert result["expired"] is True
        assert result["changes"] is None
        assert result["history_id"] == "77777"
        assert "full sync" in result["guidance"]

    @pytest.mark.asyncio
    async def test_other_http_errors_are_not_swallowed(self):
        service = _build_service([])
        service.users.return_value.history.return_value.list.return_value.execute = (
            MagicMock(side_effect=_http_error(500))
        )
        with pytest.raises(HttpError):
            await _unwrap(get_gmail_history)(
                service=service,
                user_google_email="erik@example.com",
                start_history_id="100",
            )


class TestPageBudget:
    @pytest.mark.asyncio
    async def test_hitting_the_budget_withholds_the_checkpoint_and_warns(self):
        """A truncated walk must not let the caller advance its checkpoint,
        which would skip every change it did not see."""
        pages = [
            {
                "history": [{"id": "1", "messagesAdded": [_msg("m1", "t1")]}],
                "historyId": "500",
                "nextPageToken": "p2",
            },
            {
                "history": [{"id": "2", "messagesAdded": [_msg("m2", "t2")]}],
                "historyId": "500",
                "nextPageToken": "p3",
            },
        ]
        service = _build_service(pages)
        result = await _unwrap(get_gmail_history)(
            service=service,
            user_google_email="erik@example.com",
            start_history_id="100",
            max_pages=2,
        )
        assert result["complete"] is False
        assert result["history_id"] is None
        assert "do NOT advance" in result["warning"]
        assert result["pages"] == 2
        # The partial changes are still returned, just marked partial.
        assert result["thread_ids"] == ["t1", "t2"]


class TestValidation:
    @pytest.mark.asyncio
    async def test_unknown_history_type_is_rejected_with_the_valid_list(self):
        service = _build_service([{"historyId": "500"}])
        with pytest.raises(UserInputError) as excinfo:
            await _unwrap(get_gmail_history)(
                service=service,
                user_google_email="erik@example.com",
                start_history_id="100",
                history_types=["messageAdded", "somethingElse"],
            )
        message = str(excinfo.value)
        assert "somethingElse" in message
        assert "labelRemoved" in message

    @pytest.mark.asyncio
    async def test_validation_runs_before_any_api_call(self):
        service = _build_service([{"historyId": "500"}])
        with pytest.raises(UserInputError):
            await _unwrap(get_gmail_history)(
                service=service,
                user_google_email="erik@example.com",
                start_history_id="100",
                history_types=["nope"],
            )
        service.users.return_value.history.return_value.list.assert_not_called()
