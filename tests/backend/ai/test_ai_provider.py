"""Exercise offline replay and HTTP errors without contacting a model service."""
import io
import json
from urllib.error import HTTPError, URLError

import pytest

from backend.ai.provider import OfflineProvider, OpenAICompatibleProvider, ProviderError


MESSAGES = [{"role": "system", "content": "Return candidate JSON"},
            {"role": "user", "content": "schedule this public DAG"}]


class Response(io.BytesIO):
    status = 200


def completion(text='{"schedule": []}', **extra):
    return json.dumps({"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 4}, **extra}).encode()


def test_offline_file_replays_raw_candidate_and_usage_envelope(tmp_path):
    path = tmp_path / "responses.json"
    path.write_text(json.dumps({"responses": [
        {"schedule": ["op1", "op2"]},
        {"text": '{"schedule": ["op2", "op1"]}', "usage": {"total_tokens": 12}},
    ]}))
    provider = OfflineProvider(path)
    first = provider.generate(MESSAGES)
    assert json.loads(first["text"]) == {"schedule": ["op1", "op2"]}
    assert first["usage"] == {}
    assert provider.generate(MESSAGES)["usage"] == {"total_tokens": 12}
    with pytest.raises(ProviderError, match="exhausted"):
        provider.generate(MESSAGES)


def test_offline_errors_are_never_treated_as_success(tmp_path):
    path = tmp_path / "responses.json"
    path.write_text("{broken")
    with pytest.raises(ProviderError, match="offline response JSON"):
        OfflineProvider(path)
    with pytest.raises(ProviderError, match="byte limit"):
        OfflineProvider(["é" * 10], max_response_bytes=15).generate(MESSAGES)
    with pytest.raises(ProviderError, match="must contain text"):
        OfflineProvider([{"text": ""}]).generate(MESSAGES)
    with pytest.raises(ProviderError, match="JSON values"):
        OfflineProvider([{"score": float("nan")}]).generate(MESSAGES)


@pytest.mark.parametrize("content", [
    '{"responses":[{"schedule":[],"schedule":["n0"]}]}',
    '{"responses":[{"text":"candidate","usage":{"total_tokens":NaN}}]}',
])
def test_offline_queue_cannot_normalize_ambiguous_or_nonfinite_json(tmp_path, content):
    path = tmp_path/"responses.json"
    path.write_text(content)
    with pytest.raises(ProviderError, match="offline response JSON"):
        OfflineProvider(path)


def test_http_transport_gets_configured_request_and_returns_raw_text(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_KEY", "test-credential")
    requests = []

    def transport(request, *, timeout):
        requests.append((request, timeout))
        return Response(completion())

    provider = OpenAICompatibleProvider(model="test-model", base_url="https://example.test/v1/",
                                       api_key_env="TEST_MODEL_KEY", timeout=7, transport=transport)
    result = provider.generate(MESSAGES)
    request, timeout = requests[0]
    assert request.full_url == "https://example.test/v1/chat/completions"
    assert request.get_method() == "POST" and timeout == 7
    assert request.get_header("Authorization") == "Bearer test-credential"
    assert json.loads(request.data) == {"model": "test-model", "messages": MESSAGES, "stream": False}
    assert json.loads(result["text"]) == {"schedule": []}
    assert result["usage"]["completion_tokens"] == 4


@pytest.mark.parametrize("failure", [
    HTTPError("https://example.test", 401, "credential-in-error-body", {}, None),
    URLError("credential-in-error-body"), TimeoutError("credential-in-error-body"),
])
def test_http_failures_do_not_expose_server_errors_or_credentials(monkeypatch, failure):
    monkeypatch.setenv("TEST_MODEL_KEY", "test-credential")

    def transport(request, *, timeout):
        raise failure

    provider = OpenAICompatibleProvider(model="test", base_url="https://example.test/v1",
                                       api_key_env="TEST_MODEL_KEY", transport=transport)
    with pytest.raises(ProviderError) as error:
        provider.generate(MESSAGES)
    assert "test-credential" not in str(error.value)
    assert "credential-in-error-body" not in str(error.value)


@pytest.mark.parametrize("body,diagnostic", [
    (b"not JSON", "invalid chat completion"),
    (b'{"error": {"message": "server error"}}', "invalid chat completion"),
    (completion(None), "invalid chat completion"),
    (b'{"choices":[],"choices":[{"message":{"content":"{}"}}]}', "invalid chat completion"),
    (b'{"choices":[{"message":{"content":"partial"},"finish_reason":"length"}]}', "truncated"),
    (b"x" * 257, "byte limit"),
])
def test_malformed_truncated_and_oversized_http_responses_fail(monkeypatch, body, diagnostic):
    monkeypatch.setenv("TEST_MODEL_KEY", "test-credential")
    provider = OpenAICompatibleProvider(model="test", base_url="https://example.test/v1",
                                       api_key_env="TEST_MODEL_KEY", max_response_bytes=256,
                                       transport=lambda request, **kwargs: Response(body))
    with pytest.raises(ProviderError, match=diagnostic):
        provider.generate(MESSAGES)


def test_missing_key_fails_before_request_and_embedded_url_credentials_are_rejected(monkeypatch):
    monkeypatch.delenv("MISSING_TEST_MODEL_KEY", raising=False)

    def no_network(*args, **kwargs):
        pytest.fail("request must not run without configured credentials")

    provider = OpenAICompatibleProvider(model="test", base_url="http://localhost:1234/v1",
                                       api_key_env="MISSING_TEST_MODEL_KEY", transport=no_network)
    with pytest.raises(ProviderError, match="environment variable"):
        provider.generate(MESSAGES)
    with pytest.raises(ProviderError, match="embedded credentials"):
        OpenAICompatibleProvider(model="test", base_url="https://user:secret@example.test/v1")


def test_default_http_transport_does_not_forward_authorization_on_redirect():
    provider = OpenAICompatibleProvider(model="test", base_url="https://example.test/v1")
    redirect = next(handler for handler in provider.transport.__self__.handlers
                    if hasattr(handler, "redirect_request"))
    assert redirect.redirect_request(None, None, 302, "Found", {}, "https://other.test") is None
