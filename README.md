# EmbeddingGemma OpenVINO Benchmark

C++ and Python benchmarks for **Google's EmbeddingGemma-300m** embedding model using OpenVINO™ 2026.0. Compares **INT4 vs INT8 vs FP32** precision on **CPU and GPU**, measuring performance (latency) and accuracy (Recall@1, cosine similarity).

## Features

- **Python benchmark** (`embedding_gemma_benchmark.py`) — recommended, supports INT4/INT8/FP32 with GPU reshape optimization
- **C++ benchmark** (`embedding_gemma_benchmark`) — raw OpenVINO C++ API, handles `position_ids` manually
- **GenAI TextEmbeddingPipeline test** (`genai_embedding_test`) — uses the high-level `ov::genai::TextEmbeddingPipeline` API
- **GPU reshape optimization** — fully static `[1, N]` reshape for INT4/INT8 on GPU eliminates kernel recompilation
- **10 built-in test scenarios** covering astronomy, history, CS, biology, geography, physics, music, medicine, technology, and environment
- **External TSV dataset support** — load custom retrieval datasets (STS-B included)
- **Automated Recall@1** accuracy metric with per-scenario reporting
- **Cross-precision cosine similarity** comparison (INT4/INT8 vs FP32 baseline)

## Prerequisites

### 1. OpenVINO™ 2026.0

Download and extract the **OpenVINO GenAI** package for Windows x64:

- [OpenVINO GenAI 2026.0 Downloads](https://docs.openvino.ai/2026/get-started/install-openvino.html)

Or use the pre-built archive:
```
openvino_genai_windows_2026.0.0.0_x86_64/
├── setupvars.ps1          # Environment setup script
├── runtime/               # OpenVINO runtime + GenAI headers/libs
├── python/                # Python bindings
└── samples/               # Sample applications
```

### 2. Visual Studio 2022

Install **Visual Studio 2022** with the following workloads:
- Desktop development with C++
- CMake tools for Windows

Or install standalone: [CMake 3.15+](https://cmake.org/download/) and [MSVC Build Tools](https://visualstudio.microsoft.com/downloads/).

### 3. EmbeddingGemma-300m Models

Export the model in FP32, INT8, and INT4 formats using [optimum-intel](https://github.com/huggingface/optimum-intel):

```bash
# Install dependencies
pip install optimum-intel[openvino] nncf

# Export FP32
optimum-cli export openvino --model google/embeddinggemma-300m --task feature-extraction --weight-format fp32 --trust-remote-code embeddinggemma-ov-optimum\fp32

# Export INT8
optimum-cli export openvino --model google/embeddinggemma-300m --task feature-extraction --weight-format int8 --trust-remote-code embeddinggemma-ov-optimum\int8

# Export INT4 (NNCF quantization)
optimum-cli export openvino --model google/embeddinggemma-300m --task feature-extraction --weight-format int4 --trust-remote-code embeddinggemma-ov-optimum\int4
```

**Model sizes:**

| Precision | Size |
|-----------|------|
| FP32 | 1,155 MB |
| INT8 | 290 MB |
| INT4 | 244 MB |

Each exported directory should contain:
```
openvino_model.xml / openvino_model.bin   # Model IR
openvino_tokenizer.xml / .bin             # Tokenizer
openvino_detokenizer.xml / .bin           # Detokenizer
tokenizer_config.json, config.json, etc.
```

## Setup Environment

Open PowerShell and run the OpenVINO setup script:

```powershell
# Set up OpenVINO environment variables
& "C:\path\to\openvino_genai_windows_2026.0.0.0_x86_64\setupvars.ps1"
```

This sets `OpenVINO_DIR`, `OpenVINOGenAI_DIR`, and adds runtime DLLs to `PATH`.

## Build

```powershell
# Navigate to the project directory
cd embedding-gemma-openvino

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build Release
cmake --build . --config Release
```

This produces two executables in `build/Release/`:
- `embedding_gemma_benchmark.exe` — Raw OV API benchmark (recommended)
- `genai_embedding_test.exe` — GenAI SDK API test

## Run

### Python Benchmark (recommended)

```powershell
# Set up OpenVINO environment
& "C:\path\to\openvino_genai_windows_2026.0.0.0_x86_64\setupvars.ps1"

# Basic usage — all precisions, all devices
python embedding_gemma_benchmark.py `
    "C:\path\to\embeddinggemma-ov-optimum\int4" `
    "C:\path\to\embeddinggemma-ov-optimum\fp32"

# With INT8 model
python embedding_gemma_benchmark.py `
    "C:\path\to\embeddinggemma-ov-optimum\int4" `
    "C:\path\to\embeddinggemma-ov-optimum\fp32" `
    --int8-dir "C:\path\to\embeddinggemma-ov-optimum\int8"

# GPU only, all precisions
python embedding_gemma_benchmark.py `
    "C:\path\to\embeddinggemma-ov-optimum\int4" `
    "C:\path\to\embeddinggemma-ov-optimum\fp32" `
    --int8-dir "C:\path\to\embeddinggemma-ov-optimum\int8" `
    --device GPU

# GPU only, single precision, custom iterations
python embedding_gemma_benchmark.py `
    "C:\path\to\embeddinggemma-ov-optimum\int4" `
    "C:\path\to\embeddinggemma-ov-optimum\fp32" `
    --int8-dir "C:\path\to\embeddinggemma-ov-optimum\int8" `
    --precision INT8 --device GPU --warmup 2 --iters 30

# With external TSV dataset and custom seq-len
python embedding_gemma_benchmark.py `
    "C:\path\to\embeddinggemma-ov-optimum\int4" `
    "C:\path\to\embeddinggemma-ov-optimum\fp32" `
    --dataset stsb_retrieval.tsv --seq-len 512
```

**Python CLI Arguments:**

| Argument | Description | Default |
|----------|-------------|----------|
| `int4_dir` | Path to INT4 model directory | (required) |
| `fp32_dir` | Path to FP32 model directory | (required) |
| `--int8-dir` | Path to INT8 model directory | (optional) |
| `--device` | Device: `CPU`, `GPU`, or `ALL` | `ALL` |
| `--precision` | Precision: `INT4`, `INT8`, `FP32`, or `ALL` | `ALL` |
| `--warmup` | Warmup iterations | `3` |
| `--iters` | Benchmark iterations | `10` |
| `--seq-len` | Fixed sequence length for GPU reshape | `256` |
| `--dataset` | External TSV dataset file | built-in |

### C++ Benchmark

```powershell
# Basic usage with built-in 10 scenarios
.\Release\embedding_gemma_benchmark.exe `
    "C:\path\to\embedding-gemma-int4-ov" `
    "C:\path\to\embeddinggemma-ov-optimum\fp32" `
    3 10

# With external TSV dataset
.\Release\embedding_gemma_benchmark.exe `
    "C:\path\to\embedding-gemma-int4-ov" `
    "C:\path\to\embeddinggemma-ov-optimum\fp32" `
    3 10 stsb_retrieval.tsv
```

**Arguments:**
| Arg | Description | Default |
|-----|-------------|---------|
| `INT4_MODEL_DIR` | Path to INT4 model directory | (required) |
| `FP32_MODEL_DIR` | Path to FP32 model directory | (required) |
| `warmup` | Warmup iterations | 3 |
| `iters` | Benchmark iterations | 10 |
| `dataset.tsv` | External TSV dataset file | built-in |

### GenAI TextEmbeddingPipeline Test

```powershell
.\Release\genai_embedding_test.exe `
    "C:\path\to\embedding-gemma-int4-ov" `
    "C:\path\to\embeddinggemma-ov-optimum\fp32" `
    CPU 2 5
```

> **Note:** `TextEmbeddingPipeline` currently does not set `position_ids`, which EmbeddingGemma requires. Tests will report `SKIP` until the GenAI SDK adds this support. Use `embedding_gemma_benchmark.exe` instead.

## TSV Dataset Format

The external dataset file uses tab-separated values:

```
query	expected_idx	doc_0	doc_1	doc_2	doc_3
What is the capital of France?	1	Berlin is the capital of Germany.	Paris is the capital of France.	Madrid is the capital of Spain.	Rome is the capital of Italy.
```

- **Header row** is required (skipped on load)
- `expected_idx` is the 0-based index of the correct document
- Supports 2+ documents per row

### Included Datasets

| File | Scenarios | Source |
|------|-----------|--------|
| `test_dataset.tsv` | 20 | Hand-crafted diverse topics |
| `stsb_retrieval.tsv` | 50 | STS-B (high-sim pairs + random negatives) |
| `stsb_test.tsv` | 1,379 | Full STS-B test set (sentence pairs + scores) |

## Test Results

### Environment

| Component | Version |
|-----------|---------|
| CPU | Intel® Core™ Ultra (Series 2) |
| GPU | Intel® Arc™ B580 |
| OS | Windows 11 |
| OpenVINO | 2026.0.0 |
| Model | EmbeddingGemma-300m (768-dim, 24 layers) |

### GPU Benchmark (Python, 30 iterations, 10 built-in scenarios)

```
python embedding_gemma_benchmark.py int4 fp32 --int8-dir int8 --device GPU --warmup 2 --iters 30

================================================================
  Performance & Accuracy Summary
================================================================
Config              Avg(ms)     Min(ms)     Max(ms)     Iters   Recall@1
------------------------------------------------------------------------------
FP32 / GPU          17.31       13.83       73.01       600     10/10 (100%)
INT8 / GPU          6.99        6.57        20.00       1500    10/10 (100%)
INT4 / GPU          6.74        6.37        8.21        1500    10/10 (100%)
------------------------------------------------------------------------------
```

> INT4/INT8 on GPU use fully static reshape `[1, 256]` by default. Each text is embedded individually (batch=1), matching real RAG pipeline usage. This eliminates GPU kernel recompilation and yields **2.5x speedup** over FP32.

### INT4/INT8 vs FP32 Accuracy (Cosine Similarity)

| Precision | Mean Query Cosine Sim | Mean Doc Cosine Sim |
|-----------|----------------------|---------------------|
| INT8 | 0.9998 | 0.9997 |
| INT4 | 0.9870 | 0.9844 |

> INT8 quantization preserves **>99.97% embedding fidelity** with 4x size reduction (1155MB → 290MB).
> INT4 quantization preserves **>98.4% embedding fidelity** with 4.7x size reduction (1155MB → 244MB).
> **All configurations achieve 100% Recall@1** on the built-in test set.

### Key Findings

- **INT4 GPU is the fastest** overall (6.74ms avg) — 2.6x faster than FP32 GPU
- **INT8 GPU is nearly as fast** (6.99ms avg) with near-lossless accuracy (cosine sim >0.999)
- **INT8 is the best accuracy/size tradeoff** — 4x smaller than FP32 with negligible quality loss
- **GPU reshape `[1, N]`** is critical for quantized models on GPU (eliminates kernel recompilation)
- **All configurations achieve 100% Recall@1** on the 10 built-in scenarios

## Architecture Notes

### Why Raw OV API Instead of GenAI TextEmbeddingPipeline?

EmbeddingGemma uses a `Gemma3TextModel` architecture with **bidirectional attention** and requires `position_ids` as a model input. The GenAI `TextEmbeddingPipeline` ([source](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/cpp/src/rag/text_embedding_pipeline.cpp)) sets `input_ids` and `attention_mask` but does **not** set `position_ids`, causing a MatMul shape mismatch error at inference time.

The `TextRerankPipeline` in the same codebase correctly handles `position_ids` — this is a known gap in the GenAI SDK.

### GPU Reshape Optimization

Quantized models (INT4/INT8) on GPU suffer from **kernel recompilation** when input tensor shapes change between inference calls. This is because quantized graphs have ~40% more operations (dequantization nodes), and each new shape triggers OpenCL/L0 kernel rebuilds.

The benchmark uses `model.reshape({...: [1, seq_len]})` to set a **fully static shape** (batch=1, fixed sequence length) for all model inputs. This:
- Eliminates **all** kernel recompilation — kernels compile once and are reused
- Uses **batch=1** which matches real RAG usage (documents indexed one-at-a-time, queries embedded individually)
- Pads/truncates all inputs to the fixed `--seq-len` (default: 256 tokens)

The `--seq-len` parameter can be tuned: use `64` for short queries, `256` for typical RAG chunks, `512` for longer documents (max: 2048 per model's `max_position_embeddings`).

FP32 on GPU does **not** use reshape because it has fewer ops and handles dynamic shapes more efficiently.

### Embedding Pipeline

```
Input Text → SentenceTransformer Prompt → Tokenize → Model Inference → Mean Pooling → L2 Normalize → Embedding Vector (768-dim)
```

- **Query prompt:** `"task: search result | query: <text>"`
- **Document prompt:** `"title: none | text: <text>"`
- **Pooling:** Attention-mask-aware mean pooling over token dimension
- **Normalization:** L2 normalization to unit vectors
- **Similarity:** Cosine similarity for ranking

## License

Apache License 2.0 — See [LICENSE](LICENSE) for details.

## Related

- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)
- [EmbeddingGemma on Hugging Face](https://huggingface.co/google/embedding-gemma-300m)
- [optimum-intel](https://github.com/huggingface/optimum-intel)
