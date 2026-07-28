#!/usr/bin/env python3
"""Run a disposable TLS reverse proxy and certify Laplace remote boundaries."""

from __future__ import annotations

import argparse
import http.client
import http.server
import json
import os
import secrets
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Sequence

from capture_user_guide_screenshots import (
    ADMIN_EMAIL,
    _application,
    _free_port,
    _server,
)


ROOT = Path(__file__).resolve().parents[1]
_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class _ThreadedServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _proxy_handler(backend_port: int) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _forward(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 2_000_000:
                self.send_error(413)
                return
            body = self.rfile.read(length) if length else None
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in _HOP_HEADERS
            }
            headers["Host"] = self.headers.get("Host", "")
            headers["X-Forwarded-Host"] = self.headers.get("Host", "")
            headers["X-Forwarded-Proto"] = "https"
            headers["X-Forwarded-For"] = self.client_address[0]
            connection = http.client.HTTPConnection(
                "127.0.0.1",
                backend_port,
                timeout=20,
            )
            try:
                connection.request(self.command, self.path, body=body, headers=headers)
                response = connection.getresponse()
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() not in _HOP_HEADERS:
                        self.send_header(key, value)
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = response.read(16_384)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            finally:
                connection.close()
                self.close_connection = True

        do_GET = _forward
        do_POST = _forward
        do_PATCH = _forward
        do_DELETE = _forward
        do_OPTIONS = _forward

        def log_message(self, _format: str, *_arguments: object) -> None:
            return

    return Handler


def _redirect_handler(target: str) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(308)
            self.send_header("Location", f"{target}{self.path}")
            self.end_headers()

        def log_message(self, _format: str, *_arguments: object) -> None:
            return

    return Handler


def _serve(server: _ThreadedServer) -> tuple[threading.Thread, _ThreadedServer]:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread, server


def _certificate(root: Path) -> tuple[Path, Path]:
    certificate = root / "fixture-certificate.pem"
    key = root / "fixture-key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-nodes",
            "-days",
            "30",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    os.chmod(key, 0o600)
    return certificate, key


def _closed_ports(count: int = 2) -> list[int]:
    listeners: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(count):
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            ports.append(int(listener.getsockname()[1]))
            listeners.append(listener)
    finally:
        for listener in listeners:
            listener.close()
    return ports


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    backend_port = _free_port()
    tls_port = _free_port()
    redirect_port = _free_port()
    external_url = f"https://localhost:{tls_port}"
    password = f"remote-fixture-{secrets.token_urlsafe(32)}"
    result: dict[str, object]

    with tempfile.TemporaryDirectory(prefix="laplace-remote-fixture-") as temporary:
        fixture_root = Path(temporary)
        certificate, key = _certificate(fixture_root)
        app = _application(
            fixture_root / "state",
            backend_port,
            password,
            f"plus-{secrets.token_urlsafe(32)}",
            external_url=external_url,
        )
        proxy = _ThreadedServer(
            ("127.0.0.1", tls_port),
            _proxy_handler(backend_port),
        )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certificate, key)
        proxy.socket = context.wrap_socket(proxy.socket, server_side=True)
        redirect = _ThreadedServer(
            ("127.0.0.1", redirect_port),
            _redirect_handler(external_url),
        )
        proxy_thread, _ = _serve(proxy)
        redirect_thread, _ = _serve(redirect)
        try:
            for _operator_server in _server(app, backend_port):
                redirect_connection = http.client.HTTPConnection(
                    "127.0.0.1",
                    redirect_port,
                    timeout=5,
                )
                redirect_connection.request("GET", "/")
                redirect_response = redirect_connection.getresponse()
                redirect_result = {
                    "status": "PASS"
                    if redirect_response.status == 308
                    and redirect_response.getheader("Location") == f"{external_url}/"
                    else "FAIL",
                    "http_status": redirect_response.status,
                }
                redirect_response.read()
                redirect_connection.close()

                model_ports = _closed_ports()
                command = [
                    sys.executable,
                    str(ROOT / "scripts/check_remote_access.py"),
                    "--external-url",
                    external_url,
                    "--expected-host",
                    "localhost",
                    "--ca-file",
                    str(certificate),
                    "--email",
                    ADMIN_EMAIL,
                    "--password-stdin",
                ]
                for port in model_ports:
                    command.extend(["--model-port", str(port)])
                environment = dict(os.environ)
                environment["PYTHONPATH"] = str(ROOT / "src")
                completed = subprocess.run(
                    command,
                    input=f"{password}\n",
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=60,
                    env=environment,
                )
                try:
                    checker: object = json.loads(completed.stdout)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("remote checker did not emit JSON") from exc
                if not isinstance(checker, dict):
                    raise RuntimeError("remote checker emitted an invalid result")
                result = {
                    "schema_version": 1,
                    "status": (
                        "PASS"
                        if completed.returncode == 0
                        and checker.get("status") == "PASS"
                        and redirect_result["status"] == "PASS"
                        else "FAIL"
                    ),
                    "backend_bind": "127.0.0.1",
                    "reverse_proxy_fixture": "stdlib TLS streaming proxy",
                    "https_redirect": redirect_result,
                    "checker": checker,
                    "model_endpoint_fixture_ports": model_ports,
                    "contains_credentials": False,
                }
        finally:
            proxy.shutdown()
            redirect.shutdown()
            proxy.server_close()
            redirect.server_close()
            proxy_thread.join(timeout=10)
            redirect_thread.join(timeout=10)

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
