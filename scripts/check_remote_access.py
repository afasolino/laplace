#!/usr/bin/env python3
"""Self-check remote Laplace HTTPS, policy boundaries, and private model ports."""

from __future__ import annotations

import argparse
import getpass
import http.cookiejar
import json
import socket
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-url", required=True)
    parser.add_argument("--expected-host")
    parser.add_argument("--ca-file")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--email", help="Optional registered email for cookie/logout checks")
    parser.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read the optional login password from stdin without echo",
    )
    parser.add_argument(
        "--allow-http-loopback",
        action="store_true",
        help="Permit HTTP only for a loopback reverse-proxy fixture",
    )
    parser.add_argument("--model-port", action="append", type=int, default=[8102, 8103])
    return parser


def _request(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: object | None = None,
    timeout: float,
) -> tuple[int, dict[str, str], bytes]:
    data = (
        json.dumps(body, separators=(",", ":")).encode("utf-8")
        if body is not None
        else None
    )
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        response = opener.open(request, timeout=timeout)
        return (
            response.status,
            {key.lower(): value for key, value in response.headers.items()},
            response.read(),
        )
    except urllib.error.HTTPError as exc:
        return (
            exc.code,
            {key.lower(): value for key, value in exc.headers.items()},
            exc.read(),
        )


def _tls_result(
    host: str,
    port: int,
    *,
    ca_file: str | None,
    timeout: float,
) -> dict[str, object]:
    context = ssl.create_default_context(cafile=ca_file)
    with socket.create_connection((host, port), timeout=timeout) as raw:
        with context.wrap_socket(raw, server_hostname=host) as secure:
            certificate = secure.getpeercert()
            if certificate is None:
                raise RuntimeError("TLS peer did not provide a certificate")
            expires_raw = str(certificate.get("notAfter", ""))
            expires = ssl.cert_time_to_seconds(expires_raw)
            remaining = int((datetime.fromtimestamp(expires, UTC) - datetime.now(UTC)).total_seconds() // 86400)
            return {
                "status": "PASS" if remaining >= 14 else "FAIL",
                "protocol": secure.version(),
                "hostname_verified": True,
                "expires_at": datetime.fromtimestamp(expires, UTC).isoformat(),
                "days_remaining": remaining,
            }


def _port_private(host: str, port: int, timeout: float) -> dict[str, object]:
    try:
        with socket.create_connection((host, port), timeout=min(timeout, 2.0)):
            return {"port": port, "status": "FAIL_EXTERNALLY_REACHABLE"}
    except OSError as exc:
        return {
            "port": port,
            "status": "PASS_NOT_REACHABLE",
            "evidence": type(exc).__name__,
        }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    parsed = urllib.parse.urlsplit(arguments.external_url.rstrip("/"))
    expected_host = arguments.expected_host or parsed.hostname
    loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if (
        not parsed.hostname
        or parsed.scheme not in {"http", "https"}
        or (parsed.scheme != "https" and not (arguments.allow_http_loopback and loopback))
    ):
        print(json.dumps({"status": "FAIL", "failure_category": "invalid_external_url"}))
        return 2
    if expected_host != parsed.hostname:
        print(json.dumps({"status": "FAIL", "failure_category": "hostname_mismatch"}))
        return 2

    context = ssl.create_default_context(cafile=arguments.ca_file)
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cookie_jar),
        urllib.request.HTTPSHandler(context=context),
    )
    base = arguments.external_url.rstrip("/")
    results: dict[str, object] = {
        "external_url": base,
        "scheme": {
            "status": "PASS",
            "value": parsed.scheme,
            "fixture_http": parsed.scheme == "http",
        },
    }
    if parsed.scheme == "https":
        results["tls"] = _tls_result(
            parsed.hostname,
            parsed.port or 443,
            ca_file=arguments.ca_file,
            timeout=arguments.timeout,
        )
    else:
        results["tls"] = {"status": "FIXTURE_HTTP_ONLY"}

    page_status, page_headers, page_body = _request(
        opener, f"{base}/", timeout=arguments.timeout
    )
    page_text = page_body.decode("utf-8", errors="replace")
    results["login_page"] = {
        "status": "PASS"
        if page_status == 200
        and 'type="email"' in page_text
        and "password_hash" not in page_text
        and "afasolino@unisa.it" not in page_text
        else "FAIL",
        "http_status": page_status,
    }
    for name, path in (("health", "/api/v1/health"), ("readiness", "/api/v1/readiness")):
        status, _headers, body = _request(opener, f"{base}{path}", timeout=arguments.timeout)
        try:
            payload: object = json.loads(body)
        except json.JSONDecodeError:
            payload = {"status": "INVALID_JSON"}
        results[name] = {"status": "PASS" if status == 200 else "FAIL", "http_status": status, "payload": payload}

    invalid_host = f"invalid-{expected_host}"
    host_status, _host_headers, _host_body = _request(
        opener,
        f"{base}/api/v1/health",
        headers={"Host": invalid_host},
        timeout=arguments.timeout,
    )
    results["explicit_host_rejection"] = {
        "status": "PASS" if host_status in {400, 421} else "FAIL",
        "http_status": host_status,
    }
    origin_status, _origin_headers, _origin_body = _request(
        opener,
        f"{base}/api/v1/auth/login",
        method="POST",
        headers={"Origin": "https://invalid.example"},
        body={"email": "nobody@example.test", "password": "invalid"},
        timeout=arguments.timeout,
    )
    results["explicit_origin_rejection"] = {
        "status": "PASS" if origin_status == 403 else "FAIL",
        "http_status": origin_status,
    }
    results["security_headers"] = {
        "status": "PASS"
        if "content-security-policy" in {key.lower() for key in page_headers}
        and page_headers.get("x-content-type-options", "").lower() == "nosniff"
        else "FAIL"
    }

    if arguments.email:
        password = (
            sys.stdin.readline().rstrip("\n")
            if arguments.password_stdin
            else getpass.getpass("Password: ")
        )
        login_status, login_headers, login_body = _request(
            opener,
            f"{base}/api/v1/auth/login",
            method="POST",
            headers={"Origin": base},
            body={"email": arguments.email, "password": password},
            timeout=arguments.timeout,
        )
        password = ""
        set_cookie = login_headers.get("set-cookie", "")
        cookie_pass = (
            login_status == 200
            and "laplace_session=" in set_cookie
            and "HttpOnly" in set_cookie
            and "SameSite=strict" in set_cookie
            and (parsed.scheme != "https" or "Secure" in set_cookie)
        )
        results["session_cookie"] = {
            "status": "PASS" if cookie_pass else "FAIL",
            "http_status": login_status,
            "secure_required": parsed.scheme == "https",
        }
        if login_status == 200:
            payload: object = json.loads(login_body)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("csrf_token"), str
            ):
                raise RuntimeError("login response omitted the CSRF token")
            stream_status, stream_headers, stream_body = _request(
                opener,
                f"{base}/api/v1/events?once=true",
                timeout=arguments.timeout,
            )
            results["streaming_through_proxy"] = {
                "status": "PASS"
                if stream_status == 200
                and stream_headers.get("content-type", "").startswith(
                    "text/event-stream"
                )
                and b"event:" in stream_body
                else "FAIL",
                "http_status": stream_status,
            }
            logout_status, _logout_headers, _logout_body = _request(
                opener,
                f"{base}/api/v1/auth/logout",
                method="POST",
                headers={"Origin": base, "X-CSRF-Token": payload["csrf_token"]},
                body={},
                timeout=arguments.timeout,
            )
            results["logout_through_proxy"] = {
                "status": "PASS" if logout_status == 200 else "FAIL",
                "http_status": logout_status,
            }
    else:
        results["session_cookie"] = {
            "status": "NOT_TESTED_CREDENTIAL_REQUIRED",
            "secure_required": parsed.scheme == "https",
        }

    model_port_results = [
        _port_private(parsed.hostname, port, arguments.timeout)
        for port in sorted(set(arguments.model_port))
    ]
    results["model_ports_private"] = model_port_results
    statuses: list[str] = []
    for value in results.values():
        if isinstance(value, dict) and isinstance(value.get("status"), str):
            statuses.append(str(value["status"]))
    statuses.extend(str(item["status"]) for item in model_port_results)
    failed = any(status.startswith("FAIL") for status in statuses)
    results["status"] = "FAIL" if failed else "PASS"
    print(json.dumps(results, indent=2, sort_keys=True))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
