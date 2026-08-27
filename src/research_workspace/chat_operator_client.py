"""Loopback client for the resident Laplace Operator.

This is deliberately a protocol adapter, not a second LaplaceCore.  It binds
only to the already-existing Operator routes:

    POST /api/v1/chat
    POST /api/v1/agent/sessions
    POST /api/v1/agent/sessions/{session_id}/run
    POST /api/v1/agent/sessions/{session_id}/messages
    POST /api/v1/agent/sessions/{session_id}/messages/async
    GET  /api/v1/agent/sessions/{session_id}/status
    GET  /api/v1/agent/sessions/{session_id}/events
    POST /api/v1/agent/sessions/{session_id}/cancel

The request schemas are read from the Operator's own OpenAPI document at
runtime.  Unknown required request fields fail closed instead of being guessed.
"""
from __future__ import annotations

import http.cookiejar
import json
import os
import re
import stat
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

JsonObject: TypeAlias = dict[str, object]
_REMOTE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RESULT_ID_RE = re.compile(r"^res_[a-f0-9]{32}$")

CHAT_PATH = "/api/v1/chat"
OPENAPI_PATH = "/api/v1/openapi.json"
SESSION_PATH = "/api/v1/session"
AGENT_CREATE_PATH = "/api/v1/agent/sessions"
AGENT_RUN_PATH = "/api/v1/agent/sessions/{session_id}/run"
AGENT_MESSAGES_PATH = "/api/v1/agent/sessions/{session_id}/messages"
AGENT_ASYNC_MESSAGES_PATH = "/api/v1/agent/sessions/{session_id}/messages/async"
AGENT_STATUS_PATH = "/api/v1/agent/sessions/{session_id}/status"
AGENT_EVENTS_PATH = "/api/v1/agent/sessions/{session_id}/events"
AGENT_CANCEL_PATH = "/api/v1/agent/sessions/{session_id}/cancel"
AGENT_RESULT_PATH = "/api/v1/agent/sessions/{session_id}/results/{result_id}"
CAPABILITIES_PATH = "/api/v1/capabilities"
PERSONAL_CORPORA_PATH = "/api/v1/personal-corpora"
PERSONAL_CORPUS_PATH = "/api/v1/personal-corpora/{corpus_id}"
AGENT_SESSIONS_PATH = "/api/v1/worktrees"

_REQUIRED_ROUTES: tuple[tuple[str, str], ...] = (
    ("post", SESSION_PATH),
    ("get", CAPABILITIES_PATH),
    ("get", AGENT_SESSIONS_PATH),
    ("post", CHAT_PATH),
    ("post", AGENT_CREATE_PATH),
    ("get", AGENT_MESSAGES_PATH),
    ("get", AGENT_STATUS_PATH),
    ("post", AGENT_CANCEL_PATH),
)


class OperatorClientError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise OperatorClientError(f"operator_redirect_refused:{code}:{newurl}")


@dataclass(frozen=True)
class OperatorContract:
    base_url: str
    openapi_title: str
    openapi_version: str
    chat_path: str
    agent_create_path: str
    agent_turn_path: str
    agent_async_turn_path: str | None
    agent_status_path: str
    agent_events_path: str | None
    agent_cancel_path: str
    auth_kind: str
    auth_header: str | None


@dataclass(frozen=True)
class AgentTurnResult:
    session_id: str
    payload: JsonObject


@dataclass(frozen=True)
class AgentTurnSubmission:
    session_id: str
    turn_id: str
    event_cursor: int
    payload: JsonObject


def _as_object(value: object, *, label: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise OperatorClientError(f"{label}_not_object")
    return {str(key): item for key, item in value.items()}


def _deep_get_text(value: object) -> str:
    """Extract a display answer without assuming one response envelope.

    The exact JSON object is still returned to callers; this helper is only for
    bounded terminal rendering.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        preferred = (
            "content",
            "response",
            "answer",
            "assistant",
            "text",
            "message",
            "summary",
            "result",
            "output",
        )
        for key in preferred:
            if key not in value:
                continue
            found = _deep_get_text(value[key])
            if found:
                return found
        for key in ("data", "payload"):
            if key in value:
                found = _deep_get_text(value[key])
                if found:
                    return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [_deep_get_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return ""


def extract_display_text(payload: Mapping[str, object]) -> str:
    return _deep_get_text(payload)


def _extract_session_id(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key in ("session_id", "id", "agent_session_id"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
        for key in ("binding", "session", "data", "payload", "result"):
            if key in value:
                candidate = _extract_session_id(value[key])
                if candidate:
                    return candidate
    return None


class OperatorClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8765",
        timeout_seconds: float = 30.0,
        token: str | None = None,
        opener: urllib.request.OpenerDirector | None = None,
    ) -> None:
        self.base_url = self._validate_base_url(base_url)
        self.timeout_seconds = min(max(float(timeout_seconds), 1.0), 120.0)
        self._token = self._validate_token(token) if token is not None else self._resolve_token()
        self._cookie_jar: http.cookiejar.CookieJar | None
        if opener is None:
            self._cookie_jar = http.cookiejar.CookieJar()
            self._opener = urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(self._cookie_jar),
                _NoRedirect(),
            )
        else:
            self._cookie_jar = None
            self._opener = opener
        self._openapi: JsonObject | None = None
        self._contract: OperatorContract | None = None

    @staticmethod
    def _validate_base_url(value: str) -> str:
        parsed = urllib.parse.urlsplit(value.rstrip("/"))
        if parsed.scheme != "http":
            raise OperatorClientError("operator_base_url_must_be_http_loopback")
        host = parsed.hostname
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise OperatorClientError("operator_base_url_must_be_loopback")
        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise OperatorClientError("operator_base_url_invalid")
        return value.rstrip("/")

    @staticmethod
    def _resolve_token() -> str | None:
        for name in ("LAPLACE_CHAT_TOKEN", "LAPLACE_ZETSU_TOKEN"):
            value = os.environ.get(name)
            if value:
                return OperatorClient._validate_token(value.strip())
        token_file = os.environ.get("LAPLACE_CHAT_TOKEN_FILE")
        if token_file:
            path = Path(token_file).expanduser()
            if path.is_symlink():
                raise OperatorClientError("chat_token_file_symlink_refused")
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(path, flags)
            except OSError as exc:
                raise OperatorClientError("chat_token_file_unreadable") from exc
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise OperatorClientError("chat_token_file_not_regular")
                if metadata.st_mode & 0o077:
                    raise OperatorClientError("chat_token_file_permissions_too_open")
                if metadata.st_size > 4_096:
                    raise OperatorClientError("chat_token_file_too_large")
                with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                    descriptor = -1
                    value = handle.read(4_097).strip()
            except (OSError, UnicodeDecodeError) as exc:
                raise OperatorClientError("chat_token_file_unreadable") from exc
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            if not value:
                raise OperatorClientError("chat_token_file_empty")
            return OperatorClient._validate_token(value)
        return None

    @staticmethod
    def _validate_token(value: str) -> str:
        if not value or len(value) > 4_096 or any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in value
        ):
            raise OperatorClientError("chat_token_invalid")
        return value

    def _csrf_token(self) -> str:
        """Fetch the current Operator CSRF nonce for mutation requests.

        ``POST /api/v1/session`` is the Operator's authenticated nonce-rotation
        endpoint. It intentionally does not itself require a CSRF nonce. The
        same opener is reused so any same-origin session cookie established by
        the Operator is preserved for the following mutation request.
        """
        payload = self._request(
            "POST",
            SESSION_PATH,
            authenticated=True,
            include_csrf=False,
        )
        token = payload.get("csrf_token")
        if (
            not isinstance(token, str)
            or not token
            or len(token) > 4096
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in token)
        ):
            raise OperatorClientError("operator_session_missing_valid_csrf_token")
        return token

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, object] | None = None,
        authenticated: bool = True,
        include_csrf: bool = True,
    ) -> JsonObject:
        if not path.startswith("/"):
            raise OperatorClientError("operator_path_must_be_absolute")
        url = f"{self.base_url}{path}"
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if authenticated:
            headers.update(self._auth_headers())
        if (
            authenticated
            and include_csrf
            and method.upper() in {"POST", "PUT", "PATCH", "DELETE"}
        ):
            headers["X-CSRF-Token"] = self._csrf_token()
        data = None
        if body is not None:
            data = json.dumps(
                dict(body),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method.upper(),
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise OperatorClientError("operator_response_too_large")
                if not raw:
                    return {}
                return _as_object(json.loads(raw.decode("utf-8")), label="operator_response")
        except OperatorClientError:
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read(64 * 1024).decode("utf-8", errors="replace")
            if exc.code in {401, 403} and not self._token:
                raise OperatorClientError(
                    "operator_authentication_required:"
                    "set_LAPLACE_CHAT_TOKEN_or_LAPLACE_ZETSU_TOKEN_or_LAPLACE_CHAT_TOKEN_FILE"
                ) from exc
            raise OperatorClientError(
                f"operator_http_error:{exc.code}:{detail[:4000]}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OperatorClientError(f"operator_unreachable:{exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OperatorClientError("operator_response_invalid_json") from exc

    def _load_openapi(self) -> JsonObject:
        if self._openapi is None:
            self._openapi = self._request(
                "GET",
                OPENAPI_PATH,
                authenticated=False,
            )
        return self._openapi

    def _paths(self) -> JsonObject:
        return _as_object(self._load_openapi().get("paths"), label="openapi_paths")

    def _operation(self, method: str, path: str) -> JsonObject:
        paths = self._paths()
        path_item = _as_object(paths.get(path), label=f"openapi_path:{path}")
        return _as_object(
            path_item.get(method.lower()),
            label=f"openapi_operation:{method}:{path}",
        )

    def _resolve_schema(self, schema: object) -> JsonObject:
        value = _as_object(schema, label="openapi_schema")
        ref = value.get("$ref")
        if isinstance(ref, str):
            prefix = "#/components/schemas/"
            if not ref.startswith(prefix):
                raise OperatorClientError(f"unsupported_openapi_ref:{ref}")
            name = ref[len(prefix):]
            components = _as_object(
                self._load_openapi().get("components"),
                label="openapi_components",
            )
            schemas = _as_object(components.get("schemas"), label="openapi_schemas")
            return self._resolve_schema(schemas.get(name))
        all_of = value.get("allOf")
        if isinstance(all_of, list):
            merged: JsonObject = {"type": "object", "properties": {}, "required": []}
            properties: dict[str, object] = {}
            required: list[str] = []
            for part in all_of:
                resolved = self._resolve_schema(part)
                part_props = resolved.get("properties")
                if isinstance(part_props, Mapping):
                    properties.update({str(k): v for k, v in part_props.items()})
                part_required = resolved.get("required")
                if isinstance(part_required, list):
                    required.extend(str(item) for item in part_required)
            merged["properties"] = properties
            merged["required"] = sorted(set(required))
            return merged
        return value

    def _request_schema(self, method: str, path: str) -> JsonObject | None:
        operation = self._operation(method, path)
        request_body = operation.get("requestBody")
        if request_body is None:
            return None
        body_obj = _as_object(request_body, label="openapi_request_body")
        if "$ref" in body_obj:
            body_obj = self._resolve_schema(body_obj)
        content = _as_object(body_obj.get("content"), label="openapi_request_content")
        media = content.get("application/json")
        if media is None:
            raise OperatorClientError(
                f"operator_json_request_not_supported:{method}:{path}"
            )
        media_obj = _as_object(media, label="openapi_json_media")
        schema = media_obj.get("schema")
        if schema is None:
            return None
        return self._resolve_schema(schema)

    @staticmethod
    def _candidate(
        candidates: Mapping[str, object],
        property_name: str,
    ) -> tuple[bool, object]:
        if property_name in candidates and candidates[property_name] is not None:
            return True, candidates[property_name]
        return False, None

    def _build_body(
        self,
        method: str,
        path: str,
        candidates: Mapping[str, object],
    ) -> JsonObject | None:
        schema = self._request_schema(method, path)
        if schema is None:
            return None
        properties_raw = schema.get("properties")
        if not isinstance(properties_raw, Mapping):
            raise OperatorClientError(
                f"unsupported_request_schema:{method}:{path}:non_object"
            )
        properties = {str(k): v for k, v in properties_raw.items()}
        required_raw = schema.get("required")
        required = (
            {str(item) for item in required_raw}
            if isinstance(required_raw, list)
            else set()
        )
        result: JsonObject = {}
        missing: list[str] = []
        for name, prop_schema in properties.items():
            found, candidate = self._candidate(candidates, name)
            if found:
                result[name] = candidate
                continue
            if name in required:
                resolved_prop = (
                    self._resolve_schema(prop_schema)
                    if isinstance(prop_schema, Mapping)
                    else {}
                )
                if "default" not in resolved_prop:
                    missing.append(name)
        if missing:
            raise OperatorClientError(
                f"operator_contract_missing_required_inputs:{method}:{path}:"
                + ",".join(sorted(missing))
            )
        return result

    def _auth_headers(self) -> dict[str, str]:
        if self._contract is None:
            # OpenAPI itself is fetched unauthenticated; parse auth lazily here.
            self.contract_check()
        assert self._contract is not None
        if not self._token:
            return {}
        if self._contract.auth_kind == "http_bearer":
            return {"Authorization": f"Bearer {self._token}"}
        if self._contract.auth_kind == "api_key" and self._contract.auth_header:
            return {self._contract.auth_header: self._token}
        if self._contract.auth_kind == "none":
            # Existing Laplace deployments historically used bearer auth even
            # when a custom dependency was not represented in OpenAPI.
            return {"Authorization": f"Bearer {self._token}"}
        raise OperatorClientError("operator_auth_scheme_unsupported")

    def _detect_auth(self) -> tuple[str, str | None]:
        openapi = self._load_openapi()
        components = openapi.get("components")
        if not isinstance(components, Mapping):
            return ("none", None)
        schemes = components.get("securitySchemes")
        if not isinstance(schemes, Mapping) or not schemes:
            return ("none", None)
        supported: list[tuple[str, str | None]] = []
        for raw in schemes.values():
            if not isinstance(raw, Mapping):
                continue
            kind = raw.get("type")
            if kind == "http" and str(raw.get("scheme", "")).lower() == "bearer":
                supported.append(("http_bearer", "Authorization"))
            elif kind == "apiKey" and raw.get("in") == "header" and isinstance(raw.get("name"), str):
                supported.append(("api_key", str(raw["name"])))
        unique = list(dict.fromkeys(supported))
        if len(unique) == 1:
            return unique[0]
        if not unique:
            return ("none", None)
        raise OperatorClientError("operator_multiple_supported_auth_schemes_ambiguous")

    def contract_check(self) -> OperatorContract:
        paths = self._paths()
        for method, path in _REQUIRED_ROUTES:
            item = paths.get(path)
            if not isinstance(item, Mapping) or method not in item:
                raise OperatorClientError(f"operator_required_route_missing:{method}:{path}")

        if (
            isinstance(paths.get(AGENT_MESSAGES_PATH), Mapping)
            and "post" in _as_object(
                paths.get(AGENT_MESSAGES_PATH), label="openapi_agent_messages_path"
            )
        ):
            turn_path = AGENT_MESSAGES_PATH
        elif (
            isinstance(paths.get(AGENT_RUN_PATH), Mapping)
            and "post" in _as_object(
                paths.get(AGENT_RUN_PATH), label="openapi_agent_run_path"
            )
        ):
            turn_path = AGENT_RUN_PATH
        else:
            raise OperatorClientError("operator_agent_turn_route_missing")

        auth_kind, auth_header = self._detect_auth()
        info = self._load_openapi().get("info")
        info_obj = _as_object(info, label="openapi_info") if isinstance(info, Mapping) else {}
        contract = OperatorContract(
            base_url=self.base_url,
            openapi_title=str(info_obj.get("title", "")),
            openapi_version=str(info_obj.get("version", "")),
            chat_path=CHAT_PATH,
            agent_create_path=AGENT_CREATE_PATH,
            agent_turn_path=turn_path,
            agent_async_turn_path=(
                AGENT_ASYNC_MESSAGES_PATH
                if isinstance(paths.get(AGENT_ASYNC_MESSAGES_PATH), Mapping)
                and "post" in _as_object(
                    paths.get(AGENT_ASYNC_MESSAGES_PATH),
                    label="openapi_agent_async_messages_path",
                )
                else None
            ),
            agent_status_path=AGENT_STATUS_PATH,
            agent_events_path=(
                AGENT_EVENTS_PATH
                if isinstance(paths.get(AGENT_EVENTS_PATH), Mapping)
                and "get" in _as_object(
                    paths.get(AGENT_EVENTS_PATH),
                    label="openapi_agent_events_path",
                )
                else None
            ),
            agent_cancel_path=AGENT_CANCEL_PATH,
            auth_kind=auth_kind,
            auth_header=auth_header,
        )
        self._contract = contract
        return contract

    @staticmethod
    def _format_path(template: str, session_id: str) -> str:
        if _REMOTE_ID_RE.fullmatch(session_id) is None:
            raise OperatorClientError("invalid_remote_session_id")
        return template.replace("{session_id}", urllib.parse.quote(session_id, safe=""))

    def chat(
        self,
        *,
        lane: str,
        messages: Sequence[Mapping[str, str]],
        domain: str,
        session_id: str | None,
        retrieval_selection: str = "none",
        personal_corpus_id: str | None = None,
    ) -> JsonObject:
        body = self._build_body(
            "post",
            CHAT_PATH,
            {
                "lane": lane,
                "messages": [dict(message) for message in messages],
                "domain": domain,
                "session_id": session_id,
                "retrieval_selection": retrieval_selection,
                "personal_corpus_id": personal_corpus_id,
            },
        )
        return self._request("POST", CHAT_PATH, body=body)

    def capabilities(self) -> JsonObject:
        """Return deterministic owner capabilities and authorized repository IDs."""

        return self._request("GET", CAPABILITIES_PATH)
    def personal_corpora(self, *, include_archived: bool = False) -> JsonObject:
        query = urllib.parse.urlencode({"include_archived": str(include_archived).lower()})
        return self._request("GET", f"{PERSONAL_CORPORA_PATH}?{query}")

    def personal_corpus(self, corpus_id: str) -> JsonObject:
        if re.fullmatch(r"pc_[a-f0-9]{32}", corpus_id) is None:
            raise OperatorClientError("invalid_personal_corpus_id")
        path = PERSONAL_CORPUS_PATH.replace(
            "{corpus_id}", urllib.parse.quote(corpus_id, safe="")
        )
        return self._request("GET", path)

    def create_agent_session(
        self,
        *,
        repo_id: str,
        requested_session_id: str,
        task_title: str,
    ) -> AgentTurnResult:
        body = self._build_body(
            "post",
            AGENT_CREATE_PATH,
            {
                "repo_id": repo_id,
                "session_id": requested_session_id,
                "task_title": task_title,
                "idempotency_key": f"chat:{requested_session_id}",
            },
        )
        payload = self._request("POST", AGENT_CREATE_PATH, body=body)
        session_id = _extract_session_id(payload)
        if not session_id:
            raise OperatorClientError("agent_create_response_missing_session_id")
        return AgentTurnResult(session_id=session_id, payload=payload)

    def run_agent_turn(
        self,
        *,
        session_id: str,
        instruction: str,
        lane: str,
        domain: str,
        retrieval_selection: str = "none",
        personal_corpus_id: str | None = None,
        max_steps: int | None = None,
        max_chars: int | None = None,
        verification_argv: Sequence[str] | None = None,
        wait_timeout_seconds: int | None = None,
    ) -> JsonObject:
        contract = self.contract_check()
        path = self._format_path(contract.agent_turn_path, session_id)
        body = self._build_body(
            "post",
            contract.agent_turn_path,
            {
                "instruction": instruction,
                "lane": lane,
                "domain": domain,
                "retrieval_selection": retrieval_selection,
                "personal_corpus_id": personal_corpus_id,
                "max_steps": max_steps,
                "max_chars": max_chars,
                "verification_argv": (
                    list(verification_argv) if verification_argv is not None else None
                ),
                "wait_timeout_seconds": wait_timeout_seconds,
            },
        )
        return self._request("POST", path, body=body)

    def submit_agent_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        instruction: str,
        lane: str,
        domain: str,
        retrieval_selection: str = "none",
        personal_corpus_id: str | None = None,
        max_steps: int | None = None,
        max_chars: int | None = None,
        verification_argv: Sequence[str] | None = None,
        wait_timeout_seconds: int | None = None,
    ) -> AgentTurnSubmission | None:
        """Submit one idempotent durable turn when the Operator exposes it.

        ``None`` preserves compatibility with a pre-async Operator, whose
        existing synchronous ``/messages`` route remains the caller fallback.
        """

        contract = self.contract_check()
        if contract.agent_async_turn_path is None:
            return None
        if _REMOTE_ID_RE.fullmatch(turn_id) is None:
            raise OperatorClientError("invalid_agent_turn_id")
        path = self._format_path(contract.agent_async_turn_path, session_id)
        body = self._build_body(
            "post",
            contract.agent_async_turn_path,
            {
                "turn_id": turn_id,
                "instruction": instruction,
                "lane": lane,
                "domain": domain,
                "retrieval_selection": retrieval_selection,
                "personal_corpus_id": personal_corpus_id,
                "max_steps": max_steps,
                "max_chars": max_chars,
                "verification_argv": (
                    list(verification_argv) if verification_argv is not None else None
                ),
                "wait_timeout_seconds": wait_timeout_seconds,
            },
        )
        payload = self._request("POST", path, body=body)
        returned_session = _extract_session_id(payload)
        returned_turn = payload.get("turn_id")
        cursor = payload.get("event_cursor")
        if returned_session != session_id or returned_turn != turn_id:
            raise OperatorClientError("agent_async_submission_identity_invalid")
        if not isinstance(cursor, int) or cursor < 0:
            raise OperatorClientError("agent_async_submission_cursor_invalid")
        return AgentTurnSubmission(
            session_id=session_id,
            turn_id=turn_id,
            event_cursor=cursor,
            payload=payload,
        )

    def agent_status(self, session_id: str) -> JsonObject:
        path = self._format_path(AGENT_STATUS_PATH, session_id)
        return self._request("GET", path)

    def agent_messages(self, session_id: str) -> JsonObject:
        path = self._format_path(AGENT_MESSAGES_PATH, session_id)
        return self._request("GET", path)

    def agent_events(
        self,
        session_id: str,
        *,
        after_sequence: int,
    ) -> list[JsonObject]:
        """Read one bounded SSE page, reconnecting with the durable sequence cursor."""

        if not 0 <= after_sequence <= 2**63 - 1:
            raise OperatorClientError("invalid_agent_event_cursor")
        contract = self.contract_check()
        if contract.agent_events_path is None:
            raise OperatorClientError("operator_agent_events_route_missing")
        path = self._format_path(contract.agent_events_path, session_id)
        query = urllib.parse.urlencode({"after_sequence": after_sequence, "once": "true"})
        headers = {"Accept": "text/event-stream", "Last-Event-ID": str(after_sequence)}
        headers.update(self._auth_headers())
        request = urllib.request.Request(
            f"{self.base_url}{path}?{query}",
            headers=headers,
            method="GET",
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise OperatorClientError("operator_event_response_too_large")
        except OperatorClientError:
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read(64 * 1024).decode("utf-8", errors="replace")
            raise OperatorClientError(f"operator_http_error:{exc.code}:{detail[:4000]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OperatorClientError(f"operator_unreachable:{exc}") from exc
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OperatorClientError("operator_event_response_invalid") from exc
        records: list[JsonObject] = []
        for block in text.split("\n\n"):
            lines = block.splitlines()
            if not any(line == "event: agent_event" for line in lines):
                continue
            data = next((line[5:].strip() for line in lines if line.startswith("data:")), None)
            if data is None:
                raise OperatorClientError("operator_event_response_invalid")
            try:
                records.append(_as_object(json.loads(data), label="operator_agent_event"))
            except json.JSONDecodeError as exc:
                raise OperatorClientError("operator_event_response_invalid") from exc
        return records

    def agent_result_page(
        self,
        session_id: str,
        result_id: str,
        *,
        artifact: str,
        offset: int = 0,
        max_bytes: int = 24_000,
    ) -> JsonObject:
        if _RESULT_ID_RE.fullmatch(result_id) is None:
            raise OperatorClientError("invalid_result_id")
        if _REMOTE_ID_RE.fullmatch(artifact) is None or not 0 <= offset <= 2**63 - 1:
            raise OperatorClientError("invalid_result_page_request")
        if not 1 <= max_bytes <= 65_536:
            raise OperatorClientError("invalid_result_page_request")
        path = self._format_path(AGENT_RESULT_PATH, session_id).replace(
            "{result_id}", urllib.parse.quote(result_id, safe="")
        )
        query = urllib.parse.urlencode(
            {"artifact": artifact, "offset": offset, "max_bytes": max_bytes}
        )
        return self._request("GET", f"{path}?{query}")

    def list_agent_sessions(self) -> JsonObject:
        return self._request("GET", AGENT_SESSIONS_PATH)

    def cancel_agent(self, session_id: str) -> JsonObject:
        path = self._format_path(AGENT_CANCEL_PATH, session_id)
        body = self._build_body("post", AGENT_CANCEL_PATH, {})
        return self._request("POST", path, body=body)

    def status_view(self, payload: Mapping[str, object], key: str) -> object:
        """Read structured evidence from status without invoking any model."""
        aliases: dict[str, tuple[str, ...]] = {
            "diff": ("diff", "patch", "git_diff", "diff_summary", "diff_hash"),
            "tests": (
                "tests",
                "verification",
                "verifier",
                "verification_result",
                "verification_summary",
            ),
            "state": ("state", "status"),
            "result": ("result", "summary", "message", "output"),
        }
        for name in aliases.get(key, (key,)):
            if name in payload:
                return payload[name]
        for container in (
            "last_result",
            "worktree_status",
            "response",
            "data",
            "payload",
            "result",
        ):
            nested = payload.get(container)
            if isinstance(nested, Mapping):
                for name in aliases.get(key, (key,)):
                    if name in nested:
                        return nested[name]
        return {"status": "not_present_in_agent_status", "field": key}
