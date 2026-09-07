([English](README.md)|简体中文|[日本語](README_ja.md)|[한국어](README_ko.md))

# FunASR OpenAI 兼容 API 服务

FunASR OpenAI 兼容 API 提供 `/v1/audio/transcriptions`，可作为私有语音转写服务接入 OpenAI 风格 SDK 或 multipart HTTP 客户端。它实现的是语音接口子集，不是完整 OpenAI API，也不保证兼容所有 SDK/框架功能。

本页启动仓库中的[示例服务](server.py)。打包的 `funasr-server` 是[另一套实现](../../funasr/bin/_server_app.py)，复用配置前请先看[接口边界](#api-contract)。如果需要进程内 `AutoModel.generate()` 而非 HTTP 请求，请查看 [Python SDK 指南](../../docs/python_api_zh.md)（[English](../../docs/python_api.md)）。

## API Contract

| 入口 | 启动预加载模型 | 省略 multipart `model` 时 | 说话人开关 |
|---|---|---|---|
| 在本目录运行 `python server.py` | `sensevoice` | `sensevoice` | 没有 `spk` 表单字段，只保留模型本身已经返回的说话人标签。 |
| `funasr-server` | `--model auto`：设备字符串以 `cuda` 开头时选 `fun-asr-nano`，否则选 `sensevoice` | `fun-asr-nano` | `spk=true` 为非原生说话人模型请求单独的说话人处理，默认 `False`。 |

每次请求都应明确指定 `model`：启动预加载与请求默认值是不同设置。请查询实际服务的 `/v1/models`；例如 `paraformer-en` 在示例服务中注册，却不是打包服务的内置别名。表单字段应以运行中服务的 `/openapi.json` 核对，不能只依赖仓库中的[示例规范](OPENAPI_zh.md)。

`response_format=verbose_json` 只选择响应格式，**不会启用说话人分离，也不会强制生成时间戳**。此示例仅在模型返回 `sentence_info` 时将其转换为 `segments`，否则返回 `segments=[]`。说话人标签可能缺失或为 null。MOSS 原生输出匿名标签，不需要 `spk=true` 或外部 VAD/CAM++。

此示例的 `duration` 是 `generate()` 调用的耗时，单位为秒，不包含初次模型加载，**不是音频时长**。打包服务的 verbose 响应使用秒单位的音频时长，其 fallback 在无法读取音频元数据时可能使用 0。两套服务的片段 `start`/`end` 均为秒。打包服务的 fallback 可能根据文本与音频时长生成粗粒度片段，并非逐词强制对齐。打包服务的 verbose 结构包含 `task` 及片段 `id`/`words`，示例服务则包含 `model`，不能假设 JSON 字段完全相同。参见[响应示例与说话人请求](CLIENTS.md#api-contract)。

## 快速开始

从仓库根目录执行：

```bash
pip install funasr fastapi uvicorn python-multipart
cd examples/openai_api
python server.py --model sensevoice --device cuda --port 8000
```

模型加载后再检查 `GET /health`；下载及启动耗时取决于 checkpoint、缓存、网络和硬件。除非另有说明，下文命令均在本目录执行。

需要直接复制的接入示例？可以继续查看 [客户端配方](CLIENTS.md)、[JavaScript/TypeScript 配方](JAVASCRIPT_zh.md)、[Gradio 浏览器 Demo](GRADIO_zh.md)、[工作流配方](WORKFLOWS_zh.md)、[Postman 集合](POSTMAN_zh.md)、[OpenAPI 规范](OPENAPI_zh.md)、[安全与网关指南](SECURITY_zh.md) 和 [Kubernetes 部署模板](kubernetes/README_zh.md)。

### 端到端 smoke test

在另一个终端运行：

```bash
bash smoke_test.sh
# 不依赖 curl/bash 的跨平台方式：
python smoke_test.py
```

等价手动命令：

```bash
curl -L https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav
curl http://localhost:8000/health
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

## Gradio 浏览器 Demo

如果希望用本地浏览器上传音频或测试麦克风，先启动 API 服务，再运行可选 Gradio 前端：

```bash
pip install gradio
python gradio_app.py --base-url http://localhost:8000
```

这个浏览器 demo 调用的就是 smoke test 使用的 OpenAI 兼容 API 端点。Docker、Kubernetes 和生产注意事项见 [Gradio 浏览器 Demo](GRADIO_zh.md)。

## 使用 OpenAI SDK

先运行 `pip install openai` 安装独立的 HTTP 客户端；它不是 FunASR Python SDK。

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

result = client.audio.transcriptions.create(
    model="sensevoice",  # 也可以使用 "paraformer"、"paraformer-en"、"fun-asr-nano"
    file=open("meeting.wav", "rb"),
)
print(result.text)

verbose = client.audio.transcriptions.create(
    model="sensevoice",
    file=open("meeting.wav", "rb"),
    response_format="verbose_json",
)
# verbose_json does not enable diarization; segments may be empty
print(getattr(verbose, "segments", []))
```

## 使用 curl

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=sensevoice

curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

## 可用模型

下列别名来自此示例的 `MODEL_CONFIGS`，不是整个 SDK 或所有服务统一支持的模型列表。接口会去除返回文本中的 `<|...|>` 富文本标签，不会输出独立的情感/事件字段。

| Model | 示例配置 | 响应限制 |
|---|---|---|
| `sensevoice` | SenseVoiceSmall + FSMN-VAD | 默认不启用句子时间戳或外部说话人聚类。 |
| `paraformer` | `paraformer-zh` + FSMN-VAD + CT 标点 | 已配置标点；仅设置 `verbose_json` 不会请求句子记录。 |
| `paraformer-en` | `paraformer-en` + FSMN-VAD | 相对于打包服务，这是示例专有别名；此处没有配置标点组件。 |
| `fun-asr-nano` | `AutoModel` 加载 Fun-ASR-Nano，HF 平台 + FSMN-VAD | 此示例不使用 vLLM；CTC 时间戳依赖完整 checkpoint 权重。 |
| `moss-transcribe-diarize` | 第三方 OpenMOSS 原生转写/说话人适配器 | 需要独立依赖环境；保留模型返回的时间戳与匿名标签。 |

具体 checkpoint 的语言与许可证信息请查看[模型选择](../../docs/model_selection_zh.md) 和模型自身许可证。FunASR 软件的 MIT 许可证不代表所有模型权重的许可证。请在实际工作负载上测量所选路线，不应把别名当作统一的速度或容量规格。

MOSS 使用固定 revision 的第三方 HF 模型，不能外挂 VAD 或说话人模型。完整的
`funasr-server`、Docker Compose、Kubernetes、vLLM、SGLang Omni、LocalAI 与
FunClip 路径见 [MOSS 部署指南](../../docs/moss_transcribe_diarize_zh.md)。

## API 端点

| Endpoint | Method | 说明 |
|---|---|---|
| `/v1/audio/transcriptions` | POST | OpenAI 兼容音频转写 |
| `/v1/models` | GET | 列出模型别名 |
| `/health` | GET | 健康检查、已加载模型和可用模型 |
| `/docs` | GET | FastAPI Swagger 文档 |

不想写代码验证接口时，可以使用 [Gradio 浏览器 Demo](GRADIO_zh.md) 做本地上传或麦克风测试，也可以导入 [Postman 集合](POSTMAN_zh.md)。如果要接入 API 网关、开发者门户或生成内部客户端，可以使用 [OpenAPI 规范](OPENAPI_zh.md)。

## Agent 与低代码工作流

可以通过 multipart HTTP 或工具函数模式接入 **LangChain**、**LlamaIndex**、**AutoGen**、**CrewAI**、**Semantic Kernel**、**Dify**、**n8n**，但应按本服务支持的字段验证具体集成。这些示例不保证兼容每个框架版本或实时 API。

- SDK、JavaScript/TypeScript 和 Agent tool 写法见 [客户端配方](CLIENTS.md) 与 [JavaScript/TypeScript 配方](JAVASCRIPT_zh.md)。
- Dify、n8n、HTTP 节点和 webhook worker 见 [工作流配方](WORKFLOWS_zh.md)。
- 图形界面 smoke test 见 [Postman 集合](POSTMAN_zh.md)。
- schema 驱动导入见 [OpenAPI 规范](OPENAPI_zh.md)。

## Docker 部署

从仓库根目录执行下列命令。默认镜像以 CPU 模式启动示例 `server.py`，不是打包的 `funasr-server`。

```bash
cd examples/openai_api
cp .env.example .env

docker compose up --build
```

等价 `docker run`：

```bash
docker build -t funasr-api .

docker run --rm -p 8000:8000 \
  -e FUNASR_DEVICE=cpu \
  -e FUNASR_MODEL=sensevoice \
  funasr-api
```

GPU 环境需要 NVIDIA Container Toolkit 和 CUDA-capable PyTorch/FunASR 镜像。适配 CUDA 依赖后，可使用：

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e FUNASR_DEVICE=cuda \
  -e FUNASR_MODEL=sensevoice \
  funasr-api
```

验证容器：

```bash
BASE_URL=http://localhost:8000 bash smoke_test.sh
python smoke_test.py --base-url http://localhost:8000
```

如果想用一个命令完成 build、run 和 smoke test，可以用 `bash validate_docker.sh` 验证便携 CPU 镜像。具备 NVIDIA Container Toolkit 且镜像已适配 CUDA 时，可以用 `bash validate_docker.sh --gpu` 以 `FUNASR_DEVICE=cuda` 跑同一套 smoke test。

## Kubernetes 部署

在跨团队共享服务或通过网关暴露服务前，请先阅读 [安全与网关指南](SECURITY_zh.md)，补齐 TLS、鉴权、上传限制、限流和日志策略。

如果需要在集群内部提供带持久化模型缓存、健康检查和私有 `ClusterIP` 的语音 API，可以从 [Kubernetes 部署模板](kubernetes/README_zh.md) 开始。先构建并推送示例镜像，应用 manifests，再通过 `kubectl port-forward` 和 `python smoke_test.py --base-url http://localhost:8000` 验证。

在没有 CUDA-capable 镜像和 GPU 调度配置前，请保持默认 CPU 模式。

## 配置

以下默认值属于示例 `server.py`，不属于 `funasr-server`；参见[接口边界](#api-contract)。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8000` | 监听端口 |
| `--device` | `cuda` | `cuda`、`cpu` 或 `mps` |
| `--model` | `sensevoice` | 启动时预加载模型 |

Docker 环境变量：

| Env | 默认值 | 说明 |
|---|---|---|
| `FUNASR_PORT` | `8000` | 传给 `server.py` 的容器端口 |
| `FUNASR_DEVICE` | `cpu` | 容器设备模式；只有在镜像已适配 CUDA 时才设为 `cuda` |
| `FUNASR_MODEL` | `sensevoice` | 容器启动时加载的模型别名 |

## 故障排查

| 现象 | 处理方式 |
|---|---|
| CUDA 不可用 | 先用 `--device cpu` 跑通 smoke test。 |
| 8000 端口被占用 | 改用 `--port 9000`，并运行 `BASE_URL=http://localhost:9000 bash smoke_test.sh` 或 `python smoke_test.py --base-url http://localhost:9000`。 |
| 模型下载很慢 | 换稳定网络，或提前从 ModelScope/Hugging Face 下载模型。 |
| Dify/n8n 容器里访问 `localhost` 失败 | 使用工作流运行时可访问的主机名、Compose service name 或 Kubernetes service name。 |
| 响应中没有 `segments` | 设置 `response_format=verbose_json` 可选择包含该字段的响应格式；数组仍可能为空，它不会启用时间戳或说话人分离。参见[接口边界](#api-contract)。 |
