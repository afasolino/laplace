"""Gradio 6 frontend for the resident Laplace Operator.

Gradio owns browser UI state/rendering.  This module is deliberately a thin
loopback client: it never creates a second LaplaceCore, model server, scheduler,
or repository sandbox.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

from .chat_operator_client import OperatorClient, OperatorClientError, extract_display_text
from .chat_session import ChatSession, ChatSessionError, ChatSessionStore
from .chat_verification import ChatVerificationError, ChatVerificationStore, resolve_verification

Message: TypeAlias = dict[str, str]
ClientFactory: TypeAlias = Callable[[str], OperatorClient]
_DEFAULT_STATE_ROOT = Path.home() / ".local" / "state" / "laplace"
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class LaplaceWebError(RuntimeError):
    pass


def _git_root(cwd: Path) -> Path:
    completed = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        raise LaplaceWebError("laplace_web_requires_git_repository")
    return Path(completed.stdout.strip()).resolve()


def _verification(value: str) -> tuple[str, ...] | None:
    if not value.strip():
        return None
    try:
        parts = tuple(shlex.split(value))
    except ValueError as exc:
        raise LaplaceWebError("invalid_verification_command") from exc
    if not parts or len(parts) > 64 or any(len(part) > 1_000 or "\x00" in part for part in parts):
        raise LaplaceWebError("invalid_verification_command")
    return parts


def _default_client(url: str) -> OperatorClient:
    return OperatorClient(base_url=url)


class WebController:
    def __init__(
        self,
        *,
        repository_root: Path,
        repo_id: str,
        state_root: Path,
        operator_url: str,
        client_factory: ClientFactory = _default_client,
    ) -> None:
        if _REPO_ID_RE.fullmatch(repo_id) is None:
            raise LaplaceWebError("invalid_repo_id")
        self.repository_root = repository_root.resolve()
        self.repo_id = repo_id
        self.operator_url = operator_url
        self.store = ChatSessionStore(state_root.resolve() / "chat_sessions_v2")
        self.verifiers = ChatVerificationStore(state_root.resolve() / "chat_verifiers_v3")
        self.client_factory = client_factory

    def _client(self) -> OperatorClient:
        client = self.client_factory(self.operator_url)
        client.contract_check()
        return client

    def _load_or_create(
        self,
        session_id: str,
        *,
        mode: str,
        access: str,
        lane: str,
    ) -> ChatSession:
        if session_id:
            session = self.store.load(
                session_id,
                repo_id=self.repo_id,
                repository_root=str(self.repository_root),
            )
            if session.remote_agent_session_id is not None and lane != session.lane:
                raise LaplaceWebError("agent_session_lane_is_pinned:start_new_session")
            return self.store.update(
                session,
                interaction_mode=mode,
                access_mode=access,
                lane=lane,
            )
        return self.store.create(
            repo_id=self.repo_id,
            repository_root=str(self.repository_root),
            lane=lane,
            domain="software_engineering",
            interaction_mode=mode,
            access_mode=access,
        )

    def capabilities(self) -> str:
        from .chat_discovery import render_capabilities

        return render_capabilities(self._client().capabilities())

    def new_session(self) -> tuple[str, list[Message], str]:
        return "", [], "New browser session will be created on the next message."

    def resume(self, requested: str) -> tuple[str, list[Message], str]:
        value = requested.strip()
        if value == "last":
            session = self.store.last(
                repo_id=self.repo_id,
                repository_root=str(self.repository_root),
            )
            if session is None:
                raise LaplaceWebError("no_previous_session")
        else:
            session = self.store.load(
                value,
                repo_id=self.repo_id,
                repository_root=str(self.repository_root),
            )
        history = [
            {"role": item.role, "content": item.content}
            for item in session.messages
            if item.role in {"user", "assistant"}
        ]
        return session.session_id, history, f"Resumed {session.session_id}"

    def cancel(self, session_id: str) -> str:
        if not session_id:
            return "No active local session."
        session = self.store.load(
            session_id,
            repo_id=self.repo_id,
            repository_root=str(self.repository_root),
        )
        if session.remote_agent_session_id is None:
            return "No remote agent session to cancel."
        value = self._client().cancel_agent(session.remote_agent_session_id)
        return f"Cancel: {value.get('status', value)}"

    def respond(
        self,
        message: str,
        history: Sequence[Mapping[str, object]] | None,
        session_id: str,
        mode: str,
        access: str,
        lane: str,
        verification_text: str,
    ) -> tuple[list[Message], str, str, str]:
        text = message.strip()
        current: list[Message] = [
            {"role": str(item.get("role")), "content": str(item.get("content", ""))}
            for item in (history or [])
            if item.get("role") in {"user", "assistant"}
        ]
        if not text:
            return current, session_id, "", "Message is empty."
        if mode not in {"agent", "chat"} or access not in {"read", "confirm", "write"}:
            raise LaplaceWebError("invalid_web_session_mode")
        if lane not in {"quality", "standard", "economy"}:
            raise LaplaceWebError("invalid_web_lane")

        supplied_verifier = _verification(verification_text)
        session = self._load_or_create(
            session_id,
            mode=mode,
            access=access,
            lane=lane,
        )
        try:
            verifier = resolve_verification(self.verifiers, session.session_id, supplied_verifier)
        except ChatVerificationError as exc:
            raise LaplaceWebError(str(exc)) from exc
        if access == "write" and verifier is None:
            raise LaplaceWebError("write_access_requires_verification")

        client = self._client()
        self.store.append_message(session, role="user", content=text, mode=mode)
        session = self.store.load(
            session.session_id,
            repo_id=self.repo_id,
            repository_root=str(self.repository_root),
        )
        current.append({"role": "user", "content": text})

        if mode == "chat":
            payload = client.chat(
                lane=session.lane,
                messages=[
                    {"role": item.role, "content": item.content}
                    for item in session.messages
                    if item.mode == "chat"
                ],
                domain=session.domain,
                session_id=session.session_id,
            )
        else:
            capabilities = client.capabilities()
            repositories = capabilities.get("authorized_repositories")
            authorized = {
                str(item.get("repo_id"))
                for item in repositories
                if isinstance(repositories, list) and isinstance(item, dict)
            } if isinstance(repositories, list) else set()
            if capabilities.get("agent_enabled") is not True or self.repo_id not in authorized:
                raise LaplaceWebError("repository_agent_not_authorized")
            remote = session.remote_agent_session_id
            if remote is None:
                created = client.create_agent_session(
                    repo_id=self.repo_id,
                    requested_session_id=f"web-agent-{uuid.uuid4().hex}",
                    task_title=text[:200] or "Laplace Web agent session",
                )
                remote = created.session_id
                session = self.store.update(session, remote_agent_session_id=remote)
            payload = client.run_agent_turn(
                session_id=remote,
                instruction=text,
                lane=session.lane,
                domain=session.domain,
                verification_argv=verifier,
            )

        response = extract_display_text(payload) or str(payload)
        self.store.append_message(session, role="assistant", content=response[:128_000], mode=mode)
        current.append({"role": "assistant", "content": response})
        return current, session.session_id, "", f"session={session.session_id} · mode={mode} · access={access}"


def build_app(controller: WebController):  # type: ignore[no-untyped-def]
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise LaplaceWebError("gradio_dependency_missing:install_pip_editable_with_v3_extra") from exc

    with gr.Blocks(title="Laplace") as demo:
        gr.Markdown("# Laplace\nGoverned local agent over the resident Operator")
        with gr.Row():
            mode = gr.Dropdown(["agent", "chat"], value="agent", label="Mode")
            access = gr.Dropdown(["read", "confirm", "write"], value="read", label="Access")
            lane = gr.Dropdown(["quality", "standard", "economy"], value="quality", label="Model lane")
        verifier = gr.Textbox(
            label="Deterministic verifier (required for write)",
            placeholder="pytest tests/test_example.py",
        )
        chatbot = gr.Chatbot(
            height=560,
            label="Laplace",
            layout="panel",
            buttons=["copy", "copy_all"],
            feedback_options=None,
            sanitize_html=True,
        )
        message = gr.Textbox(lines=4, label="Instruction", placeholder="Inspect the repository…")
        session_state = gr.State("")
        status = gr.Markdown("Ready. Access defaults to read-only.")
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            cancel = gr.Button("Cancel agent")
            new = gr.Button("New session")
            capabilities = gr.Button("Capabilities")
        with gr.Row():
            resume_id = gr.Textbox(label="Resume session", placeholder="last or session id")
            resume = gr.Button("Resume")

        inputs = [message, chatbot, session_state, mode, access, lane, verifier]
        outputs = [chatbot, session_state, message, status]
        send.click(controller.respond, inputs=inputs, outputs=outputs, api_name="send")
        message.submit(controller.respond, inputs=inputs, outputs=outputs, api_name="send_enter")
        cancel.click(controller.cancel, inputs=[session_state], outputs=[status], api_name="cancel")
        new.click(controller.new_session, outputs=[session_state, chatbot, status], api_name="new_session")
        capabilities.click(controller.capabilities, outputs=[status], api_name="capabilities")
        resume.click(
            controller.resume,
            inputs=[resume_id],
            outputs=[session_state, chatbot, status],
            api_name="resume",
        )
    return demo


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="laplace web")
    parser.add_argument("--repo-id")
    parser.add_argument("--operator-url", default=os.environ.get("LAPLACE_OPERATOR_URL", "http://127.0.0.1:8765"))
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.host not in {"127.0.0.1", "localhost", "::1"}:
            raise LaplaceWebError("laplace_web_loopback_only")
        root = _git_root(Path.cwd())
        repo_id = args.repo_id or os.environ.get("LAPLACE_REPO_ID") or root.name
        state_root = (
            args.state_root
            or Path(os.environ.get("LAPLACE_STATE_ROOT", str(_DEFAULT_STATE_ROOT)))
        ).expanduser().resolve()
        controller = WebController(
            repository_root=root,
            repo_id=repo_id,
            state_root=state_root,
            operator_url=args.operator_url,
        )
        controller._client()
        app = build_app(controller)
        try:
            app.queue(default_concurrency_limit=4).launch(
                server_name=args.host,
                server_port=args.port,
                inbrowser=not args.no_browser,
                share=False,
                show_error=True,
            )
        except KeyboardInterrupt:
            return 130
        return 0
    except (LaplaceWebError, ChatSessionError, OperatorClientError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
