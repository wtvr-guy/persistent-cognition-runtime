"""Streaming HTTP test adapter preserving existing transport assertions."""
from contextlib import contextmanager
import httpx


class StreamingHTTPFake:
    @contextmanager
    def stream(self, method, path, **kwargs):
        kwargs.pop("headers", None)
        result = self.post(path, **kwargs) if method == "POST" else self.get(path)
        response = httpx.Response(getattr(result, "status_code", 200), json=result.json(),
            request=httpx.Request(method, "https://ollama.test" + path))
        try:
            yield response
        finally:
            response.close()
