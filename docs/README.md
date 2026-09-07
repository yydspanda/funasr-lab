# FunASR Documentation

[中文文档](https://www.funasr.com/docs/) |
[English documentation](https://www.funasr.com/en/docs/) |
[Model Zoo](../model_zoo/readme.md) |
[Releases](https://github.com/modelscope/FunASR/releases)

## Start Here

| Your task | Guide |
| --- | --- |
| Install and check the Python environment | [Installation](installation/installation.md) / [中文](installation/installation_zh.md) |
| Choose a model, language, or timestamp capability | [Model selection](model_selection.md) / [中文](model_selection_zh.md) |
| Run the first transcription | [Quickstart](tutorial/README.md) / [中文](tutorial/README_zh.md) / [CLI](cli.md) |
| Configure AutoModel, generate, and streaming cache | [Python SDK](python_api.md) / [中文](python_api_zh.md) |
| Move from Whisper or a cloud API | [Migration](migration_from_whisper.md) / [中文](migration_from_whisper_zh.md) |

## Train and Extend

| Your task | Guide |
| --- | --- |
| Select a recipe, prepare data, train and evaluate | [Fine-tuning](training.md) / [中文](training_zh.md) |
| Add a custom architecture to AutoModel | [Model registration](model_registration.md) / [中文](model_registration_zh.md) |
| Locate supported checkpoints and individual licenses | [Model Zoo](../model_zoo/readme.md) / [中文](../model_zoo/readme_zh.md) |

## Models and Deployment

| Your task | Guide |
| --- | --- |
| Compare CPU, GPU, realtime and API runtimes | [Deployment matrix](deployment_matrix.md) / [中文](deployment_matrix_zh.md) |
| Transcribe and diarize with third-party MOSS | [MOSS](moss_transcribe_diarize.md) / [中文](moss_transcribe_diarize_zh.md) |
| Accelerate with the FunASR vLLM split engine | [vLLM](vllm_guide.md) / [中文](vllm_guide_zh.md) |
| Evaluate native vLLM serving | [Validation record](vllm_native_funasr_validation.md) |
| Deploy llama.cpp, TensorRT, Docker or Kubernetes | [Deployment manuals](https://www.funasr.com/en/deploy/) |
| Choose a service entry point and runtime SDK | [Runtime guide](../runtime/readme.md) / [中文](../runtime/readme_cn.md) |
| Select a development or serving container | [Docker](installation/docker.md) / [中文](installation/docker_zh.md) |
| Expose a Python HTTP transcription service | [Server](../examples/openai_api/README.md) / [中文](../examples/openai_api/README_zh.md) |

## Integrate Against a Contract

Python SDK, Python HTTP, native vLLM HTTP and C++ WebSocket are different
interfaces. Choose the matching protocol before selecting a client.

- [HTTP schema](../examples/openai_api/OPENAPI.md), [JavaScript](../examples/openai_api/JAVASCRIPT.md), [workflow integration](../examples/openai_api/WORKFLOWS.md)
- [C++ WebSocket protocol](../runtime/docs/websocket_protocol.md) and [ONNX binary output](../runtime/docs/onnxruntime_binary_output.md)
- [HTTP security](../examples/openai_api/SECURITY.md) and [Kubernetes manifests](../examples/openai_api/kubernetes/README.md)

## Evaluate and Operate

- [Performance methodology](benchmark/rtf_reproducibility.md) and [realtime WebSocket benchmarks](benchmark/realtime_ws_benchmark.md)
- [Troubleshooting](troubleshooting.md) / [中文](troubleshooting_zh.md)
- [Use cases](use_case_showcase.md), [community integrations](community_projects.md), and [repository responsibilities](repository_roles.md)

The product-site documentation is rendered from these Markdown sources. Edit the
source once; the site build updates article content, navigation and local search.
The catalogue lives in [documentation.json](../web-pages/product-site/data/documentation.json).
The established GitHub Pages tutorial, training and model-registration URLs are
generated from the same guides. The API reference is separately extracted from
Python source. Historical C++ release notes remain in the
[runtime archive](../runtime/release-history.md); they are not current deployment recommendations.
Model weights retain their individual licenses; the FunASR toolkit is MIT licensed.

## Build the Product Documentation

From the repository root:

```sh
python -m pip install -r web-pages/product-site/requirements-site.txt
python web-pages/product-site/build.py --output /tmp/funasr-docs
python web-pages/product-site/validate.py /tmp/funasr-docs
python scripts/gen_api_docs.py
python web-pages/product-site/export_docs.py --site /tmp/funasr-docs --output gh-pages-output
python -m http.server --directory /tmp/funasr-docs 8000
```

## Build the Sphinx Reference

The legacy Sphinx reference is a separate local build. Use an isolated environment
with the dependency versions in the `legacy-docs-links` job of
[product-site.yml](../.github/workflows/product-site.yml); installing only the
FunASR package does not reproduce that builder. From the repository root:

```sh
python -m pytest tests/test_sphinx_links.py tests/test_reference_docs_contract.py -q
python -m sphinx -b html -n -E -d /tmp/funasr-sphinx-doctrees docs /tmp/funasr-sphinx-html
```

Open `/tmp/funasr-sphinx-html/index.html` after the build. Markdown heading
fragments remain usable alongside existing Sphinx anchors, including numbered
and Chinese headings. Missing destinations still produce warnings; inspect the
build log rather than treating an exit code alone as proof of correct links.
The GitHub Pages publishing workflow uses the product documentation exporter
and generated API reference, not this standalone Sphinx output.
