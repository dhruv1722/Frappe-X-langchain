"""Deterministic confirmation handling for two-phase Sales MCP operations."""

from __future__ import annotations

import re


PREPARE_TO_CONFIRM = {
    "prepare_customer": "confirm_customer",
    "prepare_item": "confirm_item",
    "prepare_sales_order": "confirm_sales_order",
    "prepare_quotation": "confirm_quotation",
}

_CONFIRMATION_RESPONSES = frozenset({"confirm", "yes", "yes create", "create it", "go ahead"})
_CANCELLATION_RESPONSES = frozenset({"cancel", "cancel it", "do not create", "don't create", "do not confirm"})


def _normalized_reply(message: str) -> str:
    return re.sub(r"\s+", " ", message.strip().lower().replace(",", ""))


def is_confirmation_reply(message: str) -> bool:
    return _normalized_reply(message) in _CONFIRMATION_RESPONSES


def is_cancellation_reply(message: str) -> bool:
    return _normalized_reply(message) in _CANCELLATION_RESPONSES


def confirm_tool_for(prepare_tool: str) -> str | None:
    return PREPARE_TO_CONFIRM.get(prepare_tool)
