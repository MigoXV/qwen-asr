"""复制远端会议工作流，绑定本机 VAD/ASR，并发布到本机 Harness。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import ssl
import subprocess
import urllib.error
import urllib.request

import typer

app = typer.Typer(add_completion=False)
CERT = Path("/root/repositories/audio-harness/model-bin/realtime-tls/cert.pem")


def api(path, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        "https://localhost:10000" + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, context=ssl.create_default_context(cafile=str(CERT)), timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {path}: {exc.code} {exc.read().decode()}") from exc


def save(path, value):
    if path.name.endswith("-before.json") and path.exists():
        return
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


@app.command()
def main(output: Path = typer.Option(...)):
    output.mkdir(parents=True, exist_ok=True)

    def remote(path):
        # Read-only access through w223; the private reference has a self-signed certificate.
        return json.loads(subprocess.check_output([
            "ssh", "-o", "BatchMode=yes", "w223", "curl", "--noproxy", "'*'",
            "-ksS", "--max-time", "30", "https://192.168.0.222:10000" + path,
        ]))

    reference = next(w for w in remote("/api/v1/audio-workflows") if w["slug"] == "ai-meeting")
    active = remote(f"/api/v1/audio-workflows/{reference['id']}/active")
    save(output / "remote-active.json", active)
    save(output / "remote-draft.json", remote(f"/api/v1/audio-workflows/{reference['id']}/draft"))
    local_engines = api("/api/v1/engines?limit=500")["items"]
    # Never export credentials to the report directory.
    save(output / "local-engines-before.json", [
        {k: e.get(k) for k in ("id", "slug", "api", "kind", "target", "enabled")}
        for e in local_engines
    ])
    save(output / "local-workflows-before.json", api("/api/v1/audio-workflows"))

    def engine(slug, name, kind, protocol, engine_api, target):
        existing = next((e for e in local_engines if e["slug"] == slug), None)
        body = dict(slug=slug, name=name, kind=kind, protocol=protocol, api=engine_api,
                    target=target, enabled=True, request_timeout_seconds=180)
        if existing:
            for field in ("api", "protocol", "kind", "target"):
                if existing[field] != body[field]:
                    raise RuntimeError(f"Existing engine {slug} differs at {field}; preserve it")
            return existing["id"]
        return api("/api/v1/engines", "POST", body)["id"]

    vad = next(e for e in local_engines if e["slug"] == "fireredvad-ncnn")
    if not vad["enabled"]:
        raise RuntimeError("Local VAD is disabled")
    asr_id = engine("qwen-asr-1-7b-ascend", "Qwen3 ASR 1.7B Ascend fused", "asr",
                    "grpc", "ux_speech", "localhost:50011")
    dummy_id = engine("dummy-diarize", "占位说话人分离", "diarization",
                      "dummy", "dummy_diarize", "dummy://diarize")

    document = copy.deepcopy(active["document"])
    removed = {n["id"] for n in document["nodes"] if n["kind"] == "speaker_clustering"}
    document["nodes"] = [n for n in document["nodes"] if n["id"] not in removed]
    document["edges"] = [e for e in document["edges"]
                         if e["source"].get("node_id") not in removed and e["target"]["node_id"] not in removed]
    document["outputs"] = [o for o in document["outputs"] if o["source"].get("node_id") not in removed]
    document["layout"] = {k: v for k, v in document["layout"].items() if k not in removed}
    for node in document["nodes"]:
        if node["kind"] == "physical_vad":
            node["engine_id"] = vad["id"]
            node["config"]["max_segment_seconds"] = min(node["config"]["max_segment_seconds"], 30)
        elif node["kind"] == "asr":
            node["engine_id"] = asr_id
        elif node["kind"] == "speaker_diarization":
            node["engine_id"] = dummy_id
            node["config"]["max_batch_seconds"] = 30
    save(output / "local-document.json", document)
    save(output / "placeholder-policy.json", {
        "reference_revision": active["revision"], "speaker_diarization": "dummy, speaker=0",
        "speaker_clustering": "待实现；不进入执行图，不输出虚构结果",
        "max_asr_segment_seconds": 30, "remote_inference_dependencies": [],
    })
    capacity = api("/api/v1/settings/capacity")
    save(output / "capacity-before.json", capacity)
    if capacity["max_asr_concurrency"] != 32:
        body = {k: v for k, v in capacity.items() if k != "updated_at"}
        body["max_asr_concurrency"] = 32
        api("/api/v1/settings/capacity", "PUT", body)
    save(output / "capacity-after.json", api("/api/v1/settings/capacity"))
    workflow = next((w for w in api("/api/v1/audio-workflows") if w["slug"] == "ai-meeting"), None)
    if workflow is None:
        workflow = api("/api/v1/audio-workflows", "POST", {
            "slug": "ai-meeting", "name": "ai-meeting",
            "description": "本机 VAD + 融合 ASR；说话人分离占位，聚类待实现。",
            "document": document,
        })
    else:
        draft = api(f"/api/v1/audio-workflows/{workflow['id']}/draft")
        save(output / "local-draft-before.json", draft)
        if draft["document"] != document:
            api(f"/api/v1/audio-workflows/{workflow['id']}/draft", "PUT", {
                "expected_revision": draft["revision"], "document": document,
            })
    path = f"/api/v1/audio-workflows/{workflow['id']}"
    validation = api(path + "/validate", "POST")
    save(output / "validation.json", validation)
    if not validation["valid"]:
        raise RuntimeError(str(validation))
    draft = api(path + "/draft")
    published = api(path + "/publish", "POST", {"expected_revision": draft["revision"]})
    save(output / "local-active.json", published)
    models = api("/v1/models")
    save(output / "models.json", models)
    if not any(m["id"] == "ai-meeting" for m in models["data"]):
        raise RuntimeError("Published ai-meeting is missing from /v1/models")
    typer.echo(json.dumps({"workflow_id": workflow["id"], "revision": published["revision"],
                           "model": "ai-meeting", "asr": "localhost:50011",
                           "vad": vad["target"]}, ensure_ascii=False))


if __name__ == "__main__":
    app()
