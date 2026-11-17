"""Small model transports. Providers return proposals, never validation results."""
from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .protocol import parse_json


class ProviderError(RuntimeError):
    """A model transport or response failed; no response body or secret is logged."""


def _usage(value) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProviderError("model usage must be an object")
    return value


class OfflineProvider:
    """Replay JSON responses from a list or a {responses: [...]} JSON file.

    A response can be raw text, a {text, usage} envelope, or a candidate JSON
    object (serialized as text). Exhaustion is an error, never an implicit pass.
    """

    def __init__(self, responses: Iterable | str | Path, *, max_response_bytes: int = 2 * 1024 * 1024):
        if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= 16 * 1024 * 1024:
            raise ProviderError("invalid response byte limit")
        self.max_response_bytes = max_response_bytes
        if isinstance(responses, (str, Path)):
            try:
                responses = parse_json(Path(responses).read_text(encoding="utf-8"))
            except (OSError, ValueError, RecursionError) as error:
                raise ProviderError("cannot read offline response JSON") from error
        if isinstance(responses, dict) and set(responses) == {"responses"}:
            responses = responses["responses"]
        if not isinstance(responses, (list, tuple)) or not responses:
            raise ProviderError("offline provider requires a nonempty response list")
        self.responses = deque(responses)

    def generate(self, messages: list[dict]) -> dict:
        if not self.responses:
            raise ProviderError("offline response queue exhausted")
        response = self.responses.popleft()
        usage = {}
        if isinstance(response, dict) and "text" in response:
            if set(response) - {"text", "usage"}:
                raise ProviderError("invalid offline response envelope")
            text, usage = response["text"], _usage(response.get("usage"))
        elif isinstance(response, dict):
            try:
                text = json.dumps(response, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError, RecursionError):
                raise ProviderError("offline candidate must contain JSON values") from None
        else:
            text = response
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("model response must contain text")
        if len(text.encode("utf-8")) > self.max_response_bytes:
            raise ProviderError("model response exceeds byte limit")
        return {"text": text, "usage": usage}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward a bearer credential to a server-selected destination.
        return None


class OpenAICompatibleProvider:
    """Explicitly configured chat/completions transport using only stdlib.

    transport, when supplied, takes a urllib Request and timeout keyword and
    returns a context-managed byte response. No request is made at construction.
    """

    def __init__(
        self, *, model: str, base_url: str, api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60, max_response_bytes: int = 2 * 1024 * 1024,
        transport: Callable | None = None,
    ):
        if not isinstance(model, str) or not model.strip() or len(model) > 256:
            raise ProviderError("model must be explicitly configured")
        if not isinstance(base_url, str):
            raise ProviderError("base_url must be an HTTP or HTTPS URL")
        try:
            parsed = urlsplit(base_url)
            parsed.port
        except ValueError:
            raise ProviderError("base_url must be a valid HTTP or HTTPS URL") from None
        if (parsed.scheme not in {"https", "http"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or any(character.isspace() for character in base_url)):
            raise ProviderError("base_url must be HTTP(S) without embedded credentials, query or fragment")
        if not isinstance(api_key_env, str) or not api_key_env or len(api_key_env) > 128:
            raise ProviderError("api_key_env must name an environment variable")
        if not isinstance(timeout, (float, int)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ProviderError("timeout must be positive and finite")
        if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= 16 * 1024 * 1024:
            raise ProviderError("invalid response byte limit")
        endpoint = base_url.rstrip("/")
        self.endpoint = endpoint if endpoint.endswith("/chat/completions") else endpoint + "/chat/completions"
        self.model, self.api_key_env = model, api_key_env
        self.timeout, self.max_response_bytes = timeout, max_response_bytes
        self.transport = transport or build_opener(_NoRedirect()).open

    def generate(self, messages: list[dict]) -> dict:
        api_key = os.environ.get(self.api_key_env)
        if not api_key or "\n" in api_key or "\r" in api_key:
            raise ProviderError(f"missing or invalid API key environment variable: {self.api_key_env}")
        content = json.dumps({"model": self.model, "messages": messages, "stream": False}).encode("utf-8")
        request = Request(self.endpoint, data=content, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + api_key,
            "Accept": "application/json",
        })
        try:
            with self.transport(request, timeout=self.timeout) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise ProviderError(f"model request failed (HTTP {status})")
                body = response.read(self.max_response_bytes + 1)
        except HTTPError as error:
            raise ProviderError(f"model request failed (HTTP {error.code})") from None
        except (URLError, OSError, TimeoutError, ValueError):
            raise ProviderError("model request failed or timed out") from None
        if len(body) > self.max_response_bytes:
            raise ProviderError("model response exceeds byte limit")
        try:
            document = parse_json(body.decode("utf-8"))
            choices = document["choices"]
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError()
            choice = choices[0]
            if choice.get("finish_reason") in {"length", "content_filter"}:
                raise ProviderError("model response was truncated or filtered")
            text = choice["message"]["content"]
            usage = _usage(document.get("usage"))
            if not isinstance(text, str) or not text.strip():
                raise ValueError()
        except (KeyError, IndexError, TypeError, ValueError, UnicodeDecodeError, RecursionError):
            raise ProviderError("invalid chat completion response") from None
        return {"text": text, "usage": usage}
