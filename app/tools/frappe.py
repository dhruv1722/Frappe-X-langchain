import os
import json
from copy import deepcopy
from urllib.parse import quote

import requests
from langchain_core.tools import tool


FRAPPE_BASE_URL = os.environ.get("FRAPPE_BASE_URL", "").rstrip("/")
FRAPPE_API_KEY = os.environ.get("FRAPPE_API_KEY")
FRAPPE_API_SECRET = os.environ.get("FRAPPE_API_SECRET")


def _headers():
    return {
        "Authorization": f"token {FRAPPE_API_KEY}:{FRAPPE_API_SECRET}",
        "Content-Type": "application/json",
    }


def _log(title: str, **fields):
    print("\n========================================")
    print(title)
    print("========================================")
    for key, value in fields.items():
        print(f"{key}: {value}")


def _handle_response(resp) -> dict:
    print("FRAPPE STATUS:", resp.status_code)

    if resp.ok:
        try:
            data = resp.json().get("data")
        except ValueError:
            data = resp.text

        return {
            "success": True,
            "status_code": resp.status_code,
            "data": data,
        }

    print("FRAPPE RESPONSE:", resp.text)

    return {
        "success": False,
        "status_code": resp.status_code,
        "error": resp.text,
    }


class MasterLinkResolutionError(ValueError):
    """Raised when a supplied Frappe Link value cannot be resolved safely."""


def _get_doctype_fields(doctype: str) -> list[dict]:
    """Read the current DocType definition so Link targets are never guessed."""
    resp = requests.get(
        f"{FRAPPE_BASE_URL}/api/resource/DocType/{quote(doctype, safe='')}",
        headers=_headers(),
        timeout=30,
    )

    if not resp.ok:
        raise MasterLinkResolutionError(
            f"Could not validate linked master data because DocType '{doctype}' "
            "could not be read."
        )

    try:
        data = resp.json().get("data", {})
    except ValueError as exc:
        raise MasterLinkResolutionError(
            f"Could not read the DocType definition for '{doctype}'."
        ) from exc

    fields = data.get("fields", [])
    if not isinstance(fields, list):
        raise MasterLinkResolutionError(
            f"The DocType definition for '{doctype}' is invalid."
        )

    return fields


def _find_master_candidates(doctype: str, supplied_value: str) -> list[str]:
    """Return a small set of matching document names from the linked master."""
    if not supplied_value.strip():
        raise MasterLinkResolutionError(
            f"A value is required for linked master '{doctype}'."
        )

    resp = requests.get(
        f"{FRAPPE_BASE_URL}/api/resource/{quote(doctype, safe='')}",
        headers=_headers(),
        params={
            "fields": json.dumps(["name"]),
            # Frappe applies the filter server-side; the client only chooses
            # candidates and still requires one deterministic match below.
            "filters": json.dumps([["name", "like", f"%{supplied_value}%"]]),
            "limit_page_length": 10,
        },
        timeout=30,
    )

    if not resp.ok:
        raise MasterLinkResolutionError(
            f"Could not search linked master '{doctype}'."
        )

    try:
        records = resp.json().get("data", [])
    except ValueError as exc:
        raise MasterLinkResolutionError(
            f"Could not read linked master '{doctype}'."
        ) from exc

    return [
        record["name"]
        for record in records
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    ]


def _resolve_link_value(doctype: str, supplied_value: str, field_path: str) -> str:
    candidates = _find_master_candidates(doctype, supplied_value)
    supplied_key = supplied_value.casefold()
    exact_case_insensitive = [
        candidate for candidate in candidates if candidate.casefold() == supplied_key
    ]

    if len(exact_case_insensitive) == 1:
        return exact_case_insensitive[0]

    # A single substring match is safe to canonicalize. Multiple matches must
    # be resolved by the user instead of guessing which master they meant.
    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise MasterLinkResolutionError(
            f"No {doctype} master record matches '{supplied_value}' for '{field_path}'."
        )

    options = ", ".join(candidates[:5])
    raise MasterLinkResolutionError(
        f"'{supplied_value}' is ambiguous for '{field_path}'. "
        f"Matching {doctype} records: {options}."
    )


def _resolve_document_links(doctype: str, document: dict, path: str) -> list[dict]:
    resolutions: list[dict] = []

    for field in _get_doctype_fields(doctype):
        fieldname = field.get("fieldname")
        fieldtype = field.get("fieldtype")
        target_doctype = field.get("options")

        if not fieldname or fieldname not in document:
            continue

        value = document[fieldname]
        field_path = f"{path}.{fieldname}"

        if fieldtype == "Link" and isinstance(target_doctype, str) and isinstance(value, str):
            canonical_value = _resolve_link_value(target_doctype, value, field_path)
            if canonical_value != value:
                document[fieldname] = canonical_value
                resolutions.append(
                    {"field": field_path, "supplied": value, "canonical": canonical_value}
                )

        elif fieldtype == "Table" and isinstance(target_doctype, str) and isinstance(value, list):
            for index, row in enumerate(value):
                if isinstance(row, dict):
                    resolutions.extend(
                        _resolve_document_links(
                            target_doctype,
                            row,
                            f"{field_path}[{index}]",
                        )
                    )

    return resolutions


def resolve_document_master_links(doctype: str, document: dict) -> tuple[dict, list[dict]]:
    """Canonicalize static Link fields against live Frappe masters before writes.

    Dynamic Link fields are deliberately left to Frappe because their target
    DocType depends on another field and cannot be safely inferred here.
    """
    if not isinstance(document, dict):
        raise MasterLinkResolutionError("The document to validate must be an object.")

    resolved_document = deepcopy(document)
    resolutions = _resolve_document_links(doctype, resolved_document, doctype)
    return resolved_document, resolutions


# ---------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------
@tool
def frappe_create_document(doctype: str, document: dict) -> dict:
    """
    Create a new document in Frappe/ERPNext.

    Use this ONLY when the user explicitly wants to create/add/insert
    a new record. Never use this to satisfy a read/list/lookup request.

    Args:
        doctype: The Frappe DocType name, e.g. "Sales Order", "Customer".
        document: The field values for the new document, as an object.
                  Child-table fields (e.g. "items") must be a JSON array
                  of objects, not a string.
    """
    _log("FRAPPE REQUEST", DOCTYPE=doctype, DOCUMENT=document)

    resp = requests.post(
        f"{FRAPPE_BASE_URL}/api/resource/{doctype}",
        headers=_headers(),
        data=json.dumps(document),
        timeout=30,
    )

    return _handle_response(resp)


# ---------------------------------------------------------------------
# LIST / SEARCH
# ---------------------------------------------------------------------
@tool
def frappe_get_list(
    doctype: str,
    filters: list | None = None,
    fields: list[str] | None = None,
    limit: int = 20,
    order_by: str | None = None,
) -> dict:
    """
    List / search existing documents of a given DocType. Use this for any
    read-only request: "list", "show", "find", "how many", "get all", etc.

    Args:
        doctype: The Frappe DocType name, e.g. "Sales Order", "Customer".
        filters: Optional list of Frappe filter triples, e.g.
                 [["status", "=", "Open"], ["customer", "=", "Acme"]].
                 Leave empty/omit to fetch without filtering.
        fields: Optional list of fieldnames to return, e.g.
                ["name", "customer", "grand_total", "status"].
                Defaults to a small standard set if omitted.
        limit: Max number of records to return. Defaults to 20; use a
               higher number only if the user asks for more.
        order_by: Optional sort, e.g. "creation desc".
    """
    _log(
        "FRAPPE REQUEST",
        DOCTYPE=doctype,
        FILTERS=filters,
        FIELDS=fields,
        LIMIT=limit,
    )

    params = {
        "limit_page_length": limit,
    }

    if filters:
        params["filters"] = json.dumps(filters)

    if fields:
        params["fields"] = json.dumps(fields)

    if order_by:
        params["order_by"] = order_by

    resp = requests.get(
        f"{FRAPPE_BASE_URL}/api/resource/{doctype}",
        headers=_headers(),
        params=params,
        timeout=30,
    )

    return _handle_response(resp)


# ---------------------------------------------------------------------
# GET SINGLE DOCUMENT
# ---------------------------------------------------------------------
@tool
def frappe_get_document(doctype: str, name: str) -> dict:
    """
    Fetch the full detail of a single, already-identified document.

    Use this when the user refers to a specific record by its name/ID,
    e.g. "show me Sales Order SO-0004" or "what's on invoice ACC-SINV-0012".

    Args:
        doctype: The Frappe DocType name, e.g. "Sales Order".
        name: The document's unique name/ID, e.g. "SO-0004".
    """
    _log("FRAPPE REQUEST", DOCTYPE=doctype, NAME=name)

    resp = requests.get(
        f"{FRAPPE_BASE_URL}/api/resource/{doctype}/{name}",
        headers=_headers(),
        timeout=30,
    )

    return _handle_response(resp)


@tool
def frappe_get_doctype_definition(doctype: str) -> dict:
    """
    Get the actual field definition and configuration for an approved
    Frappe DocType. Use for questions such as:
    - What fields does Sales Order have?
    - Which fields are mandatory in Customer?
    - Does Sales Invoice contain an items child table?
    """

    _log("FRAPPE DOCTYPE DEFINITION REQUEST", DOCTYPE=doctype)

    resp = requests.get(
        f"{FRAPPE_BASE_URL}/api/resource/DocType/{doctype}",
        headers=_headers(),
        timeout=30,
    )

    return _handle_response(resp)

# # ---------------------------------------------------------------------
# # UPDATE
# # ---------------------------------------------------------------------
@tool
def frappe_update_document(doctype: str, name: str, document: dict) -> dict:
    """
    Update fields on an existing document.

    Use this ONLY when the user wants to change/edit/update a record that
    already exists. Never use this to create a new record.

    Args:
        doctype: The Frappe DocType name, e.g. "Sales Order".
        name: The document's unique name/ID to update, e.g. "SO-0004".
        document: Only the fields to change, as an object.
    """
    _log("FRAPPE REQUEST", DOCTYPE=doctype, NAME=name, DOCUMENT=document)

    resp = requests.put(
        f"{FRAPPE_BASE_URL}/api/resource/{doctype}/{name}",
        headers=_headers(),
        data=json.dumps(document),
        timeout=30,
    )

    return _handle_response(resp)


# # ---------------------------------------------------------------------
# # DELETE
# # ---------------------------------------------------------------------
@tool
def frappe_delete_document(doctype: str, name: str) -> dict:
    """
    Delete an existing document. Use only on an explicit, unambiguous
    delete/remove/cancel request naming a specific record.

    Args:
        doctype: The Frappe DocType name, e.g. "Sales Order".
        name: The document's unique name/ID to delete, e.g. "SO-0004".
    """
    _log("FRAPPE REQUEST", DOCTYPE=doctype, NAME=name)

    resp = requests.delete(
        f"{FRAPPE_BASE_URL}/api/resource/{doctype}/{name}",
        headers=_headers(),
        timeout=30,
    )

    return _handle_response(resp)
