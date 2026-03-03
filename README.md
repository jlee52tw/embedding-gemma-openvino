# EmbeddingGemma OpenVINO Benchmark

C++ benchmark for **Google's EmbeddingGemma-300m** embedding model using OpenVINO™ 2026.0. Compares **INT4 vs FP32** precision on **CPU and GPU**, measuring performance (TTFT, TPOT, latency percentiles) and accuracy (Recall@1, cosine similarity).

## Features

- **Raw OpenVINO C++ API benchmark** (`embedding_gemma_benchmark`) — handles `position_ids` manually for full model compatibility
- **GenAI TextEmbeddingPipeline test** (`genai_embedding_test`) — uses the high-level `ov::genai::TextEmbeddingPipeline` API
- **10 built-in test scenarios** covering astronomy, history, CS, biology, geography, physics, music, medicine, technology, and environment
- **External TSV dataset support** — load custom retrieval datasets (STS-B included)
- **Automated Recall@1** accuracy metric with per-scenario reporting
- **INT4-vs-FP32 cosine similarity** comparison to quantify quantization accuracy loss

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

Export the model in both FP32 and INT4 formats using [optimum-intel](https://github.com/huggingface/optimum-intel):

```bash
# Install dependencies
pip install optimum-intel[openvino] nncf

# Export FP32
optimum-cli export openvino \
    --model google/embedding-gemma-300m \
    --weight-format fp32 \
    --task feature-extraction \
    embeddinggemma-ov-optimum/fp32

# Export INT4 (NNCF quantization)
optimum-cli export openvino \
    --model google/embedding-gemma-300m \
    --weight-format int4 \
    --task feature-extraction \
    embedding-gemma-int4-ov
```

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

### Raw OV API Benchmark (recommended)

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
| GPU | Intel® Arc™ integrated |
| OS | Windows 11 |
| OpenVINO | 2026.0.0 |
| MSVC | 19.44 |
| Model | EmbeddingGemma-300m (768-dim, 24 layers) |

### Built-in 10 Scenarios (2 warmup, 2 iterations)

```
================================================================
  Performance & Accuracy Summary
================================================================
Config            TTFT(ms)    TPOT(ms)    p50(ms)     p99(ms)     Recall@1
--------------------------------------------------------------------------------
FP32 / CPU        16.98       43.93       36.58       90.60       10/10 (100%)
INT4 / CPU        12.25       27.74       23.93       53.56       10/10 (100%)
FP32 / GPU        14.60       20.04       18.22       63.09       10/10 (100%)
INT4 / GPU        19.25       28.84       24.54       112.02      10/10 (100%)
--------------------------------------------------------------------------------
```

### STS-B Retrieval Dataset (50 scenarios, 3 iterations)

```
================================================================
  Performance & Accuracy Summary
================================================================
Config            TTFT(ms)    TPOT(ms)    p50(ms)     p99(ms)     Recall@1
--------------------------------------------------------------------------------
FP32 / CPU        15.65       41.36       30.12       83.32       19/20 (95%)
INT4 / CPU        11.16       25.99       20.20       51.95       19/20 (95%)
FP32 / GPU        14.33       17.51       16.65       39.09       19/20 (95%)
INT4 / GPU        18.71       53.97       23.18       557.05      19/20 (95%)
--------------------------------------------------------------------------------
```

### INT4 vs FP32 Accuracy (Cosine Similarity)

| Device | Mean Query Cosine Sim | Mean Doc Cosine Sim |
|--------|----------------------|---------------------|
| CPU | 0.9875 | 0.9850 |
| GPU | 0.9875 | 0.9843 |

> INT4 quantization preserves **>98.4% embedding fidelity** compared to FP32, with **zero Recall@1 degradation** on the built-in test set.

### Key Findings

- **INT4 CPU is ~1.6x faster** than FP32 CPU (TPOT 28ms vs 44ms) with no accuracy loss
- **FP32 GPU is the fastest** overall (TPOT 20ms)
- **INT4 quantization** shows negligible accuracy degradation (cosine sim >0.98)
- **All configurations achieve 100% Recall@1** on the 10 built-in scenarios

## Architecture Notes

### Why Raw OV API Instead of GenAI TextEmbeddingPipeline?

EmbeddingGemma uses a `Gemma3TextModel` architecture with **bidirectional attention** and requires `position_ids` as a model input. The GenAI `TextEmbeddingPipeline` ([source](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/cpp/src/rag/text_embedding_pipeline.cpp)) sets `input_ids` and `attention_mask` but does **not** set `position_ids`, causing a MatMul shape mismatch error at inference time.

The `TextRerankPipeline` in the same codebase correctly handles `position_ids` — this is a known gap in the GenAI SDK.

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
