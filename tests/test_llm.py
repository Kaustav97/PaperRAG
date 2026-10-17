"""Provider request behavior through the real OpenAI client and a local endpoint."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from pydantic import BaseModel, SecretStr

from paperrag.config import Models, Settings, StageModel
from paperrag.llm import LLM


class LocalResponse(BaseModel):
    value: str


def test_stage_options_reach_provider_without_cross_stage_leaks():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            body = json.dumps({
                "id": "local-completion", "object": "chat.completion", "created": 0,
                "model": "local-model", "choices": [{
                    "index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"value": "local response"}'},
                }],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    plugins = [{"id": "fusion", "preset": "general-budget"}]
    settings = Settings(
        base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key=SecretStr("local-key"),
        models=Models(
            breakdown=StageModel(model="local-breakdown", options={"temperature": 0.1}),
            research=StageModel(model="openrouter/fusion", options={"extra_body": {"plugins": plugins}}),
            synthesis=StageModel(model="local-synthesis", options={"max_completion_tokens": 512}),
        ),
    )
    llm = LLM(settings)
    try:
        for stage in ("breakdown", "research", "synthesis"):
            result = llm.complete_json(stage, [{"role": "user", "content": "Compare these options..."}], LocalResponse)
            assert result.value == "local response"
    finally:
        llm.client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert [path for path, _ in requests] == ["/v1/chat/completions"] * 3
    breakdown, research, synthesis = [payload for _, payload in requests]
    assert [body["model"] for body in (breakdown, research, synthesis)] == [
        "local-breakdown", "openrouter/fusion", "local-synthesis",
    ]
    assert research["plugins"] == plugins
    assert "extra_body" not in research
    assert breakdown["temperature"] == 0.1
    assert synthesis["max_completion_tokens"] == 512
    assert "plugins" not in breakdown and "plugins" not in synthesis
    assert "temperature" not in research and "temperature" not in synthesis
    assert "max_completion_tokens" not in breakdown and "max_completion_tokens" not in research
    for body in (breakdown, research, synthesis):
        assert body["messages"][-1] == {"role": "user", "content": "Compare these options..."}
        assert body["messages"][0]["role"] == "system"
        assert "Return only one JSON object" in body["messages"][0]["content"]
        assert "api_key" not in body
