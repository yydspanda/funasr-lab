(English|[简体中文](README_zh.md)|[日本語](README_ja.md)|[한국어](README_ko.md))

# FunASR OpenAI-Compatible API Server

An OpenAI-style `/v1/audio/transcriptions` endpoint for private speech transcription. This example implements a speech API subset, not the entire OpenAI API or a compatibility guarantee for every SDK/framework feature.

This page starts the repository's [example server](server.py). The packaged `funasr-server` is a [different implementation](../../funasr/bin/_server_app.py); see [API boundaries](#api-contract) before reusing its settings. For in-process `AutoModel.generate()` rather than HTTP requests, use the [Python SDK guide](../../docs/python_api.md) ([中文](../../docs/python_api_zh.md)).

## API Contract

| Entry point | Startup model | Omitted multipart `model` | Speaker opt-in |
|---|---|---|---|
| `python server.py` in this directory | `sensevoice` | `sensevoice` | No `spk` form field; only preserves speaker labels already returned by the model. |
| `funasr-server` | `--model auto`: `fun-asr-nano` for a device string starting with `cuda`, otherwise `sensevoice` | `fun-asr-nano` | `spk=true` requests the separate speaker pipeline for non-native diarization models; default is `False`. |

Specify `model` explicitly in requests: startup preloading and the request default are different settings. Query the deployed `/v1/models`; for example, `paraformer-en` is registered by this example but is not a built-in alias of the packaged server. Verify fields with the running `/openapi.json`, not just the checked-in [example schema](OPENAPI.md).

`response_format=verbose_json` selects a response shape; **it does not enable diarization or force timestamp generation**. This example copies `sentence_info` into `segments` if present, otherwise returns `segments=[]`. Speaker labels can be absent or null. MOSS supplies native anonymous labels; it does not need `spk=true` or external VAD/CAM++.

In this example, `duration` is elapsed time around `generate()` in seconds, excluding initial model loading; it is **not audio duration**. The packaged server's verbose response uses audio duration in seconds (its fallback can use 0 when audio metadata is unavailable). Segment `start`/`end` use seconds in both services. The packaged fallback can synthesize coarse segments from text and audio duration; those are not word-level forced alignment. Its verbose schema includes `task` and per-segment `id`/`words`, while this example includes `model`; do not assume identical JSON fields. See [response examples and speaker requests](CLIENTS.md#api-contract).

## Quick Start

From the repository root:

```bash
pip install funasr fastapi uvicorn python-multipart
cd examples/openai_api
python server.py --model sensevoice --device cuda --port 8000
```

Wait for model loading before checking `GET /health`; download and startup time depend on the checkpoint, cache, network, and hardware. The commands below use this directory unless stated otherwise.

Need copy-paste integration snippets for Python SDK, JavaScript/TypeScript, HTTP clients, agent tools, a browser demo, Postman, OpenAPI imports, Kubernetes deployment, or Dify/n8n-style workflows? See [Client recipes](CLIENTS.md), [JavaScript/TypeScript recipes](JAVASCRIPT.md), [Gradio browser demo](GRADIO.md), [workflow recipes](WORKFLOWS.md), the [Chinese workflow recipes](WORKFLOWS_zh.md), the [Postman collection](POSTMAN.md), the [OpenAPI spec](OPENAPI.md), the [security and gateway guide](SECURITY.md), and the [Kubernetes deployment template](kubernetes/README.md).

### End-to-end smoke test

In another terminal, download a public sample and verify both health and transcription:

```bash
bash smoke_test.sh
# Cross-platform alternative without curl/bash:
python smoke_test.py
```

Equivalent manual commands:

```bash
curl -L https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav
curl http://localhost:8000/health
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

## Browser demo with Gradio

If you want a local browser UI for upload or microphone testing, run the API server first and then launch the optional Gradio frontend:

```bash
pip install gradio
python gradio_app.py --base-url http://localhost:8000
```

The browser demo calls the same OpenAI-compatible API endpoints as the smoke tests. See [Gradio browser demo](GRADIO.md) for Docker, Kubernetes, and production notes.

## Usage with OpenAI SDK (Python)

Install the separate HTTP client with `pip install openai`. This is not the FunASR Python SDK.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Basic transcription
result = client.audio.transcriptions.create(
    model="sensevoice",  # or "paraformer", "paraformer-en", "fun-asr-nano"
    file=open("meeting.wav", "rb"),
)
print(result.text)

# Inspect the verbose response; segments may be empty
result = client.audio.transcriptions.create(
    model="sensevoice",
    file=open("meeting.wav", "rb"),
    response_format="verbose_json",
)
# verbose_json does not enable diarization; see API Contract above
print(getattr(result, "segments", []))
```

## Usage with curl

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=sensevoice

# With verbose output
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

## Available Models

These are aliases in this example's `MODEL_CONFIGS`, not a universal SDK or server model list. The endpoint removes rich `<|...|>` tags from returned text; it does not expose dedicated emotion/event fields.

| Model | Example configuration | Response limits |
|---|---|---|
| `sensevoice` | SenseVoiceSmall + FSMN-VAD | Does not enable sentence timestamps or external speaker clustering by default. |
| `paraformer` | `paraformer-zh` + FSMN-VAD + CT punctuation | Punctuation is configured; `verbose_json` alone does not request sentence records. |
| `paraformer-en` | `paraformer-en` + FSMN-VAD | Example-only alias relative to the packaged server; no punctuation component configured here. |
| `fun-asr-nano` | Fun-ASR-Nano via `AutoModel`, HF hub + FSMN-VAD | Not a vLLM route in this example. CTC timestamp availability depends on complete checkpoint weights. |
| `moss-transcribe-diarize` | Third-party OpenMOSS native transcription/diarization adapter | Requires its separate dependency environment; preserves model-provided timestamps and anonymous labels. |

For checkpoint-specific language and license information, use [model selection](../../docs/model_selection.md) and the model's own license. FunASR software's MIT license is not a license for every model weight. Benchmark the selected route on your own workload; these aliases do not define universal speed or capacity.

MOSS uses a pinned third-party HF revision and must not be combined with an
external VAD or speaker model. See the [complete MOSS deployment guide](../../docs/moss_transcribe_diarize.md)
for `funasr-server`, Docker Compose, Kubernetes, vLLM, SGLang Omni, LocalAI,
and FunClip paths.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/transcriptions` | POST | Transcribe audio (OpenAI-compatible) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check + loaded models |
| `/docs` | GET | Interactive API documentation (Swagger) |

Prefer no-code API checks? Use the [Gradio browser demo](GRADIO.md) for local upload or microphone testing, or import the [Postman collection](POSTMAN.md) and run health, model-list, and transcription requests from Postman. For API gateways, developer portals, or client generation, use the [OpenAPI spec](OPENAPI.md).

## Agent Framework Integration

The multipart HTTP/tool-function pattern can be used to integrate **LangChain**, **LlamaIndex**, **AutoGen**, **CrewAI**, **Semantic Kernel**, **Dify**, and **n8n**; validate the chosen integration against this service's supported fields. These recipes do not establish compatibility with every framework version or realtime API. See [Client recipes](CLIENTS.md) and [JavaScript/TypeScript recipes](JAVASCRIPT.md) for SDK and agent-tool patterns, plus [workflow recipes](WORKFLOWS.md) for low-code HTTP nodes and webhook workers ([中文](WORKFLOWS_zh.md)).

### LangChain Example
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

def transcribe_for_agent(audio_path: str) -> str:
    """Tool function for LangChain agent."""
    result = client.audio.transcriptions.create(
        model="sensevoice", file=open(audio_path, "rb")
    )
    return result.text
```

## Docker Deployment

From the repository root, build the example image with the following commands. The default image starts the example `server.py` in CPU mode, not the packaged `funasr-server`.

```bash
cd examples/openai_api
cp .env.example .env

docker compose up --build
```

Equivalent one-off `docker run` command:

```bash
docker build -t funasr-api .

docker run --rm -p 8000:8000 \
  -e FUNASR_DEVICE=cpu \
  -e FUNASR_MODEL=sensevoice \
  funasr-api
```

For GPU hosts, use NVIDIA Container Toolkit and a CUDA-capable PyTorch/FunASR image. After adapting the image dependencies for CUDA, run the same server with `FUNASR_DEVICE=cuda`:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e FUNASR_DEVICE=cuda \
  -e FUNASR_MODEL=sensevoice \
  funasr-api
```

Verify the container from another terminal:

```bash
BASE_URL=http://localhost:8000 bash smoke_test.sh
python smoke_test.py --base-url http://localhost:8000
```

For a single build/run/smoke command, use `bash validate_docker.sh` for the portable CPU image. On a host with NVIDIA Container Toolkit and a CUDA-capable image, use `bash validate_docker.sh --gpu` to run the same smoke test with `FUNASR_DEVICE=cuda`.

## Kubernetes Deployment

Before sharing the service across a team or exposing it through a gateway, review the [security and gateway guide](SECURITY.md) for TLS, authentication, upload limits, rate limits, and logging.

For an internal cluster service with persistent model cache, health probes, and a private `ClusterIP`, start from the [Kubernetes deployment template](kubernetes/README.md). Build and push the example image, apply the manifests, then verify through `kubectl port-forward` with `python smoke_test.py --base-url http://localhost:8000`.

Keep the default CPU mode until you have built a CUDA-capable image and configured GPU scheduling for your cluster.

## Configuration

The following defaults belong to this example's `server.py`, not `funasr-server`; compare [API boundaries](#api-contract).

| Arg | Default | Description |
|-----|---------|-------------|
| `--host` | 0.0.0.0 | Bind address |
| `--port` | 8000 | Port |
| `--device` | cuda | Device (cuda/cpu/mps) |
| `--model` | sensevoice | Pre-load model at startup |

Docker environment variables:

| Env | Default | Description |
|-----|---------|-------------|
| `FUNASR_PORT` | 8000 | Container port passed to `server.py` |
| `FUNASR_DEVICE` | cpu | Container device mode; set to `cuda` only when the image has CUDA-capable dependencies |
| `FUNASR_MODEL` | sensevoice | Model alias loaded at container startup |

## Troubleshooting

- If CUDA is unavailable, use `--device cpu` for a slower but simple smoke test.
- If port 8000 is occupied, start with `--port 9000` and run `BASE_URL=http://localhost:9000 bash smoke_test.sh` or `python smoke_test.py --base-url http://localhost:9000`.
- If model download is slow, retry with a stable network or pre-download the model from ModelScope/Hugging Face.
