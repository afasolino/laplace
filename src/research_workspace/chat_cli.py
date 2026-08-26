"""Codex-like terminal front end for the resident Laplace Operator.

Ordinary chat uses /api/v1/chat.
Engineering turns use the existing durable /api/v1/agent/sessions API.
No LaplaceCore, scheduler, sandbox, corpus, or serving service is constructed in
this process.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Sequence

from .chat_operator_client import (
    OperatorClient,
    OperatorClientError,
    extract_display_text,
)
from .chat_session import ChatSession, ChatSessionError, ChatSessionStore

_CONTROL = re.compile(
    r"(?:\x1B[@-_][0-?]*[ -/]*[@-~])|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)
_MAX_RENDER = 12_000
_DEFAULT_STATE_ROOT = Path.home() / ".local" / "state" / "laplace"


class ChatCLIError(RuntimeError):
    pass


def sanitize_terminal(value: object, *, maximum: int = _MAX_RENDER) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    text = _CONTROL.sub("", text)
    if len(text) <= maximum:
        return text
    return text[: maximum - 1].rstrip() + "…"


def _git_root(cwd: Path) -> Path:
    completed = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        raise ChatCLIError("laplace_chat_requires_git_repository")
    return Path(completed.stdout.strip()).resolve()


def _repo_id(root: Path, explicit: str | None) -> str:
    value = explicit or os.environ.get("LAPLACE_REPO_ID") or root.name
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None:
        raise ChatCLIError("invalid_repo_id")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laplace chat")
    parser.add_argument("--resume", metavar="SESSION|last")
    parser.add_argument("--repo-id")
    parser.add_argument("--lane", choices=("quality", "standard", "economy"), default="quality")
    parser.add_argument("--domain", default="software_engineering")
    parser.add_argument("--mode", choices=("agent", "chat"), default="agent")
    parser.add_argument("--access", choices=("read", "confirm", "write"), default="confirm")
    parser.add_argument(
        "--operator-url",
        default=os.environ.get("LAPLACE_OPERATOR_URL", "http://127.0.0.1:8765"),
    )
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--wait-timeout-seconds", type=int)
    parser.add_argument(
        "--verification",
        nargs="+",
        help="Explicit verifier argv only when the Operator schema requires/allows it.",
    )
    return parser


class ChatShell:
    def __init__(
        self,
        *,
        client: OperatorClient,
        store: ChatSessionStore,
        session: ChatSession,
        max_steps: int | None,
        max_chars: int | None,
        wait_timeout_seconds: int | None,
        verification_argv: Sequence[str] | None,
    ) -> None:
        self.client = client
        self.store = store
        self.session = session
        self.max_steps = max_steps
        self.max_chars = max_chars
        self.wait_timeout_seconds = wait_timeout_seconds
        self.verification_argv = verification_argv

    def _require_agent_authorized(self) -> None:
        capabilities = self.client.capabilities()
        if capabilities.get("agent_enabled") is not True:
            raise ChatCLIError("agent_capability_required")
        repositories = capabilities.get("authorized_repositories")
        authorized = {
            str(item.get("repo_id"))
            for item in repositories
            if isinstance(repositories, list) and isinstance(item, dict)
        } if isinstance(repositories, list) else set()
        if self.session.repo_id not in authorized:
            choices = ",".join(sorted(authorized)) or "none"
            raise ChatCLIError(
                f"repository_not_authorized:{self.session.repo_id}:authorized_repo_ids={choices}"
            )

    def _validate_remote_session(self) -> None:
        remote = self.session.remote_agent_session_id
        if remote is None:
            return
        payload = self.client.agent_messages(remote)
        conversation = payload.get("conversation")
        if not isinstance(conversation, dict):
            raise ChatCLIError("agent_conversation_contract_invalid")
        if conversation.get("agent_session_id") != remote:
            raise ChatCLIError("agent_session_identity_mismatch")
        if conversation.get("repo_id") != self.session.repo_id:
            raise ChatCLIError("agent_session_repository_mismatch")

    def _confirm_agent_turn(self) -> None:
        self._require_agent_authorized()
        if self.verification_argv is None:
            return
        if self.session.access_mode == "read":
            raise ChatCLIError(
                "agent_mode_disabled_by_read_access:"
                "use_/access_confirm_or_/access_write_or_/mode_chat"
            )
        if self.session.access_mode == "write":
            return
        answer = input("Allow this bounded agent turn to mutate its authorized worktree? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            raise ChatCLIError("agent_turn_cancelled")

    def _ensure_remote_agent(self, *, task_title: str) -> str:
        if self.session.remote_agent_session_id:
            return self.session.remote_agent_session_id
        requested = f"chat-agent-{uuid.uuid4().hex}"
        created = self.client.create_agent_session(
            repo_id=self.session.repo_id,
            requested_session_id=requested,
            task_title=task_title.strip()[:200] or "Laplace Chat agent session",
        )
        self.session = self.store.update(
            self.session,
            remote_agent_session_id=created.session_id,
        )
        print(f"[agent session {sanitize_terminal(created.session_id, maximum=160)}]")
        return created.session_id

    def _chat_turn(self, text: str) -> None:
        self.session = self.store.append_message(
            self.session,
            role="user",
            content=text,
        )
        payload = self.client.chat(
            lane=self.session.lane,
            messages=[
                {"role": item.role, "content": item.content}
                for item in self.session.messages
                if item.mode == "chat"
            ],
            domain=self.session.domain,
            session_id=self.session.session_id,
        )
        response = extract_display_text(payload)
        if not response:
            response = sanitize_terminal(payload)
        print(sanitize_terminal(response))
        self.session = self.store.append_message(
            self.session,
            role="assistant",
            content=response[:128_000],
        )

    def _agent_turn(self, text: str) -> None:
        if self.session.active_turn_id is not None:
            raise ChatCLIError("agent_turn_active:use_/watch_or_/cancel")
        self._confirm_agent_turn()
        remote = self._ensure_remote_agent(task_title=text)
        self.session = self.store.append_message(
            self.session,
            role="user",
            content=text,
            mode="agent",
        )
        turn_id = f"turn-{uuid.uuid4().hex}"
        self.session = self.store.update(self.session, active_turn_id=turn_id)
        submit = getattr(self.client, "submit_agent_turn", None)
        submission = (
            submit(
                session_id=remote,
                turn_id=turn_id,
                instruction=text,
                lane=self.session.lane,
                domain=self.session.domain,
                max_steps=self.max_steps,
                max_chars=self.max_chars,
                verification_argv=self.verification_argv,
                wait_timeout_seconds=self.wait_timeout_seconds,
            )
            if callable(submit)
            else None
        )
        if submission is not None:
            print(f"[agent turn {sanitize_terminal(turn_id, maximum=160)} submitted]")
            try:
                rendered = self._watch_agent_turn(
                    remote,
                    turn_id,
                    after_sequence=submission.event_cursor,
                )
            except KeyboardInterrupt:
                print(sanitize_terminal(self.client.cancel_agent(remote)))
                print("[agent cancellation requested; use /watch to reconnect]")
                return
            self.session = self.store.update(self.session, active_turn_id=None)
            print(sanitize_terminal(rendered))
            self.session = self.store.append_message(
                self.session,
                role="assistant",
                content=rendered[:128_000],
                mode="agent",
            )
            return
        payload = self.client.run_agent_turn(
            session_id=remote,
            instruction=text,
            lane=self.session.lane,
            domain=self.session.domain,
            max_steps=self.max_steps,
            max_chars=self.max_chars,
            verification_argv=self.verification_argv,
            wait_timeout_seconds=self.wait_timeout_seconds,
        )
        self.session = self.store.update(self.session, active_turn_id=None)
        response = extract_display_text(payload)
        rendered = response if response else sanitize_terminal(payload)
        print(sanitize_terminal(rendered))
        self.session = self.store.append_message(
            self.session,
            role="assistant",
            content=rendered[:128_000],
            mode="agent",
        )

    def _watch_agent_turn(self, remote: str, turn_id: str, *, after_sequence: int) -> str:
        """Render durable server events; no model call is used for monitoring."""

        cursor = after_sequence
        turn_started = False
        terminal_observed = False
        terminal_wait_deadline: float | None = None
        while True:
            records = self.client.agent_events(remote, after_sequence=cursor)
            for record in records:
                sequence = record.get("sequence")
                if isinstance(sequence, int) and sequence > cursor:
                    cursor = sequence
                details = record.get("details")
                event_turn = details.get("turn_id") if isinstance(details, dict) else None
                if event_turn not in {None, turn_id}:
                    continue
                event = record.get("event")
                if isinstance(event, str):
                    print(f"[agent {sanitize_terminal(event, maximum=120)}]")
                    if event == "TURN_STARTED" and event_turn == turn_id:
                        turn_started = True
                    if event in {
                        "TURN_COMPLETED",
                        "TURN_FAILED",
                        "TURN_CANCELLED",
                        "TURN_YIELDED_RESUMABLE",
                    } and event_turn == turn_id:
                        terminal_observed = True
                    # Coordinator lifecycle events omit a UI turn ID.  Only one
                    # turn may be admitted to a remote session, and this watcher
                    # starts after TURN_SUBMITTED, so these are authoritative
                    # terminal boundaries for the current submitted turn.
                    if turn_started and event_turn is None and event in {
                        "TASK_COMPLETED",
                        "TASK_FAILED",
                        "TASK_CANCELLED",
                        "TASK_YIELDED_RESUMABLE",
                    }:
                        terminal_observed = True
            if terminal_observed:
                transcript = self.client.agent_messages(remote)
                conversation = transcript.get("conversation")
                messages = conversation.get("messages") if isinstance(conversation, dict) else None
                if isinstance(messages, list):
                    for message in reversed(messages):
                        if not isinstance(message, dict) or message.get("role") != "assistant":
                            continue
                        metadata = message.get("metadata")
                        if isinstance(metadata, dict) and metadata.get("turn_id") == turn_id:
                            content = message.get("content")
                            if isinstance(content, str) and content:
                                return content
                if terminal_wait_deadline is None:
                    terminal_wait_deadline = time.monotonic() + 15.0
                if time.monotonic() >= terminal_wait_deadline:
                    raise ChatCLIError("agent_turn_terminal_result_pending:use_/watch")
            time.sleep(0.20)

    def _require_remote(self) -> str:
        if not self.session.remote_agent_session_id:
            raise ChatCLIError("no_agent_session_in_this_chat")
        return self.session.remote_agent_session_id

    def _show_status(self, field: str | None = None) -> None:
        status = self.client.agent_status(self._require_remote())
        if field == "diff":
            worktree = status.get("worktree_status")
            result_id = worktree.get("result_id") if isinstance(worktree, dict) else None
            if isinstance(result_id, str):
                page = self.client.agent_result_page(
                    self._require_remote(),
                    result_id,
                    artifact="handoff.patch",
                )
                encoded = page.get("content_base64")
                if isinstance(encoded, str):
                    try:
                        content = base64.b64decode(encoded, validate=True).decode("utf-8")
                    except (binascii.Error, UnicodeDecodeError) as exc:
                        raise ChatCLIError("agent_result_page_invalid") from exc
                    print(sanitize_terminal(content or "No repository diff."))
                    if page.get("next_offset") is not None:
                        print(
                            f"[paged result: /result handoff.patch {page['next_offset']}]"
                        )
                    return
        value = self.client.status_view(status, field) if field else status
        print(sanitize_terminal(value))

    def command(self, line: str) -> bool:
        parts = line.strip().split()
        command = parts[0].lower()
        args = parts[1:]

        if command in {"/exit", "/quit"}:
            return False
        if command == "/help":
            print(
                "/help\n"
                "/mode agent|chat\n"
                "/access read|confirm|write\n"
                "/status\n"
                "/tasks\n"
                "/diff\n"
                "/tests\n"
                "/result <artifact> [offset]\n"
                "/cancel\n"
                "/watch\n"
                "/contract\n"
                "/context\n"
                "/history\n"
                "/compact\n"
                "/model [quality|standard|economy]\n"
                "/new\n"
                "/resume <session-id|last>\n"
                "/exit"
            )
            return True
        if command == "/mode":
            if len(args) != 1 or args[0] not in {"agent", "chat"}:
                raise ChatCLIError("usage:/mode agent|chat")
            self.session = self.store.update(self.session, interaction_mode=args[0])
            if args[0] == "agent":
                self._require_agent_authorized()
            print(f"mode={self.session.interaction_mode}")
            return True
        if command == "/access":
            if len(args) != 1 or args[0] not in {"read", "confirm", "write"}:
                raise ChatCLIError("usage:/access read|confirm|write")
            self.session = self.store.update(self.session, access_mode=args[0])
            print(f"access={self.session.access_mode}")
            return True
        if command == "/status":
            self._show_status()
            return True
        if command == "/tasks":
            print(sanitize_terminal(self.client.list_agent_sessions()))
            return True
        if command == "/diff":
            self._show_status("diff")
            return True
        if command == "/tests":
            self._show_status("tests")
            return True
        if command == "/result":
            if len(args) not in {1, 2}:
                raise ChatCLIError("usage:/result <artifact> [offset]")
            status = self.client.agent_status(self._require_remote())
            worktree = status.get("worktree_status")
            result_id = worktree.get("result_id") if isinstance(worktree, dict) else None
            if not isinstance(result_id, str):
                raise ChatCLIError("agent_result_not_available")
            try:
                offset = int(args[1]) if len(args) == 2 else 0
            except ValueError as exc:
                raise ChatCLIError("invalid_result_offset") from exc
            page = self.client.agent_result_page(
                self._require_remote(), result_id, artifact=args[0], offset=offset
            )
            print(sanitize_terminal(page))
            return True
        if command == "/cancel":
            print(sanitize_terminal(self.client.cancel_agent(self._require_remote())))
            return True
        if command == "/watch":
            if args:
                raise ChatCLIError("usage:/watch")
            if self.session.active_turn_id is None:
                raise ChatCLIError("no_active_agent_turn")
            remote = self._require_remote()
            rendered = self._watch_agent_turn(
                remote,
                self.session.active_turn_id,
                after_sequence=0,
            )
            self.session = self.store.update(self.session, active_turn_id=None)
            print(sanitize_terminal(rendered))
            return True
        if command == "/contract":
            print(sanitize_terminal(self.client.contract_check().__dict__))
            return True
        if command == "/context":
            print(
                sanitize_terminal(
                    {
                        "local_session_id": self.session.session_id,
                        "remote_agent_session_id": self.session.remote_agent_session_id,
                        "active_turn_id": self.session.active_turn_id,
                        "repo_id": self.session.repo_id,
                        "repository_root": self.session.repository_root,
                        "lane": self.session.lane,
                        "domain": self.session.domain,
                        "mode": self.session.interaction_mode,
                        "access": self.session.access_mode,
                        "chat_messages": sum(
                            item.mode == "chat" for item in self.session.messages
                        ),
                        "agent_messages": sum(
                            item.mode == "agent" for item in self.session.messages
                        ),
                    }
                )
            )
            return True
        if command == "/history":
            history: dict[str, object] = {
                "local": [
                    {"mode": item.mode, "role": item.role, "content": item.content}
                    for item in self.session.messages
                ]
            }
            if self.session.remote_agent_session_id is not None:
                history["authoritative_agent"] = self.client.agent_messages(
                    self.session.remote_agent_session_id
                )
            print(sanitize_terminal(history))
            return True
        if command == "/compact":
            if args:
                raise ChatCLIError("usage:/compact")
            before = len(self.session.messages)
            self.session = self.store.compact(self.session)
            print(f"compacted={before - len(self.session.messages)}")
            return True
        if command == "/model":
            if not args:
                print(f"lane={self.session.lane}")
                return True
            if len(args) != 1 or args[0] not in {"quality", "standard", "economy"}:
                raise ChatCLIError("usage:/model [quality|standard|economy]")
            if (
                self.session.remote_agent_session_id is not None
                and args[0] != self.session.lane
            ):
                raise ChatCLIError("agent_session_lane_is_pinned:use_/new")
            self.session = self.store.update(self.session, lane=args[0])
            print(f"lane={self.session.lane}")
            return True
        if command == "/new":
            self.session = self.store.create(
                repo_id=self.session.repo_id,
                repository_root=self.session.repository_root,
                lane=self.session.lane,
                domain=self.session.domain,
                interaction_mode=self.session.interaction_mode,
                access_mode=self.session.access_mode,
            )
            print(f"session={self.session.session_id}")
            return True
        if command == "/resume":
            if len(args) != 1:
                raise ChatCLIError("usage:/resume <session-id|last>")
            if args[0] == "last":
                found = self.store.last(
                    repo_id=self.session.repo_id,
                    repository_root=self.session.repository_root,
                )
                if found is None:
                    raise ChatCLIError("no_previous_session")
                self.session = found
            else:
                self.session = self.store.load(
                    args[0],
                    repo_id=self.session.repo_id,
                    repository_root=self.session.repository_root,
                )
            if self.session.interaction_mode == "agent":
                self._require_agent_authorized()
            self._validate_remote_session()
            print(f"session={self.session.session_id}")
            return True
        raise ChatCLIError(f"unknown_command:{command}")

    def run(self) -> int:
        contract = self.client.contract_check()
        print(
            f"Laplace | {self.session.interaction_mode} | {self.session.lane} | "
            f"repo={self.session.repo_id} | access={self.session.access_mode}"
        )
        print(
            f"Operator {contract.openapi_version or '?'} @ {contract.base_url} | "
            f"agent_turn={contract.agent_turn_path}"
        )
        if self.session.active_turn_id is not None:
            print(f"[active agent turn {sanitize_terminal(self.session.active_turn_id, maximum=160)}; use /watch]")
        while True:
            try:
                line = input("laplace> ")
            except EOFError:
                print()
                return 0
            except KeyboardInterrupt:
                print()
                continue
            try:
                if line.lstrip().startswith("/"):
                    if not self.command(line):
                        return 0
                    continue
                text = line.strip()
                if not text:
                    continue
                if self.session.interaction_mode == "chat":
                    self._chat_turn(text)
                else:
                    self._agent_turn(text)
            except (ChatCLIError, ChatSessionError, OperatorClientError) as exc:
                print(f"error: {sanitize_terminal(str(exc), maximum=4000)}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(list(argv) if argv is not None else None)
        root = _git_root(Path.cwd())
        repo_id = _repo_id(root, args.repo_id)
        state_root = (
            args.state_root
            or Path(os.environ.get("LAPLACE_STATE_ROOT", str(_DEFAULT_STATE_ROOT)))
        ).expanduser().resolve()
        store = ChatSessionStore(state_root / "chat_sessions_v2")

        if args.resume:
            if args.resume == "last":
                session = store.last(repo_id=repo_id, repository_root=str(root))
                if session is None:
                    raise ChatCLIError("no_previous_session")
            else:
                session = store.load(
                    args.resume,
                    repo_id=repo_id,
                    repository_root=str(root),
                )
        else:
            session = store.create(
                repo_id=repo_id,
                repository_root=str(root),
                lane=args.lane,
                domain=args.domain,
                interaction_mode=args.mode,
                access_mode=args.access,
            )

        client = OperatorClient(base_url=args.operator_url)
        shell = ChatShell(
            client=client,
            store=store,
            session=session,
            max_steps=args.max_steps,
            max_chars=args.max_chars,
            wait_timeout_seconds=args.wait_timeout_seconds,
            verification_argv=args.verification,
        )
        if session.interaction_mode == "agent":
            shell._require_agent_authorized()
        shell._validate_remote_session()
        return shell.run()
    except (ChatCLIError, ChatSessionError, OperatorClientError) as exc:
        print(f"error: {sanitize_terminal(str(exc), maximum=4000)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
