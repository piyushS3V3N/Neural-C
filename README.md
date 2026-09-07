# Neural-C

**A C-based Deep Learning Framework & Apple Metal GPU LLM Engine**

[![macOS](https://img.shields.io/badge/os-macOS-lightgrey.svg)](https://apple.com)
[![Language](https://img.shields.io/badge/language-C11%2FMetal-blue.svg)](https://developer.apple.com/metal/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## What We Are Building

Neural-C is a lightweight, zero-dependency deep learning framework written from scratch in C with native Apple Metal GPU acceleration. The repository contains two core components:

1. **`nn_visualizer`**: An interactive 2D graphical visualizer (built with Raylib) that renders artificial neurons, synaptic weights, loss curves, live backpropagation, Transformer multi-head attention heatmaps, an interactive **Model Select Box (dropdown menu)** to load any GGUF checkpoint in `models/` at runtime, and action buttons (`[>] RUN PROMPT`, `[>>] NEXT PRESET`, `[<>] SWITCH VISUALIZER MODE`).
2. **`neural_c_llm`**: A high-speed, quantized **Transformer LLM Inference Engine** in pure C and Metal, designed to load `.gguf` models (e.g. Qwen2.5-Coder, LLaMA-3 8B, TinyLlama, LLaMA-2) with full support for QKV projection biases, custom RoPE frequency bases (`rope_freq_base`), Tiktoken/BPE byte mapping, and dynamic GGUF special token extraction.

---

## Directory Structure

```
Neural-C/
├── Makefile                     # Build system for visualizer, LLM engine & test suite
├── README.md                    # Project overview & Quick-Start guide
├── include/                     # Public C headers
│   ├── model.h                  # Transformer hyperparameters & Config struct (QKV bias, rope_freq_base)
│   ├── tensor.h                 # Tensor & RunState activation/KV-cache buffers (bq, bk, bv bias fields)
│   ├── tokenizer.h              # BPE Tokenizer, Vocab & Special Token ID structures
│   ├── math_kernels.h           # RMSNorm, SwiGLU, RoPE, Softmax & Q4_0 MatMul
│   ├── gguf_parser.h            # GGUF binary header parser & mmap memory mapper
│   ├── llm_metal_backend.h      # Metal GPU compute bridge API (Buffer Caching)
│   ├── transformer_engine.h     # Causal Attention & Forward pipeline
│   ├── NeuralNetwork.h          # Core MLP neural network struct
│   └── visualizer.h             # Raylib 2D renderer & interactive UI state declarations
├── src/                         # Implementation source files
│   ├── tokenizer.c              # BPE tokenizer, byte-fallback & Tiktoken (Ġ/Ċ) decoder
│   ├── tensor.c                 # Tensor memory allocation routines
│   ├── gguf_parser.c            # GGUF binary parser, zero-copy mmap & special token extraction
│   ├── math_kernels.c           # SIMD & scalar quantized math kernels
│   ├── llm_metal_backend.m      # Apple Metal Objective-C GPU engine with pointer buffer caching
│   ├── transformer_engine.c     # Auto-regressive generation pipeline & QKV bias additions
│   ├── NeuralNetwork.c          # MLP CPU & GPU matrix routines
│   └── visualizer.c             # Raylib visualizer with Model Dropdown Select Box & GUI buttons
├── shaders/                     # Apple Metal MSL GPU shaders
│   ├── shaders.metal            # MLP forward pass GEMM shader
│   └── llm_shaders.metal        # Q4_0 & FP32 quantized parallel GEMV shaders
├── tests/                       # Modular test suite (Modules 1 through 7)
│   ├── test_tokenizer.c         # Module 1: Tokenizer unit test
│   ├── test_module2.c           # Module 2: Model Config & Tensor unit test
│   ├── test_module3.c           # Module 3: GGUF Parser unit test
│   ├── test_module4.c           # Module 4: Math Kernels unit test
│   ├── test_module5.c           # Module 5: Metal GPU Engine unit test
│   ├── test_module6.c           # Module 6: End-to-end LLM Engine unit test
│   └── test_module7.c           # Module 7: LLM Custom Trainer unit test
├── apps/                        # Application entry points
│   ├── main.c                   # MLP / Transformer Visualizer main GUI entry point
│   └── llm_main.c               # LLM Engine CLI entry point
├── docs/                        # Project documentation
│   ├── 01_BEGINNERS_GUIDE.md    # Educational study guide
│   ├── 02_BUILD_AND_DEPLOYMENT.md # Build & deployment engineering manual
│   └── 03_TECHNICAL_MANUAL.md   # Architectural, mathematical & low-level reference manual
└── data/                        # Datasets & persistent weight stores
    ├── dataset.csv              # NLTK names dataset
    └── brain.bin                # MLP persistent weights
```

---

## How-To Guide

### 1. Requirements (macOS)
- **Compiler**: `gcc` / `clang` (Xcode Command Line Tools)
- **GPU Framework**: Apple Metal (built-in on macOS)
- **GUI Framework (for Visualizer)**: Raylib (`brew install raylib`)

### 2. Building the Project
To compile both the LLM Engine and the Visualizer:
```bash
make all
```

To compile individual targets:
```bash
make llm         # Compiles neural_c_llm
make visualizer  # Compiles nn_visualizer
```

### 3. Running the Test Suite
Verify all 7 core engine modules:
```bash
make test
```

### 4. Running the Interactive Neural Network Visualizer
```bash
./nn_visualizer
```
- **Interactive Model Select Box**: Click the **Model Select Dropdown** in the top bar to inspect and switch between any `.gguf` checkpoint files located in the `models/` directory at runtime.
- **GUI Action Buttons**:
  - `[>] RUN PROMPT`: Trigger instant Transformer auto-regressive text generation.
  - `[>>] NEXT PRESET`: Cycle through pre-configured prompt scenarios.
  - `[<>] SWITCH VISUALIZER MODE`: Toggle between 2D Neural MLP visualization and 3D/Layered Transformer Attention View.
- **Sandbox Text Input**: Type directly into the live text prompt sandbox without performance lag.
- **MLP Mode Controls**: Use **Left / Right Arrow Keys** to cycle test dataset samples, and press **`1` (Female)** or **`0` (Male)** to trigger targeted live backpropagation.

### 5. Running the LLM Engine
```bash
# Run with default prompt on Metal GPU
./neural_c_llm

# Run with custom prompt and GGUF model file (e.g., Qwen2.5-Coder or LLaMA-3)
./neural_c_llm models/qwen2.5-coder-0.5b-instruct-q4_0.gguf "Write a C function to reverse a linked list."
```

#### How to Download Quantized GGUF Models:
You can download GGUF models (e.g. Qwen2.5-Coder 0.5B/7B, LLaMA-3 8B, TinyLlama) using `curl`:
```bash
mkdir -p models
curl -L -o models/qwen2.5-coder-0.5b-instruct-q4_0.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-q4_0.gguf"
```

Then run inference:
```bash
./neural_c_llm models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf "Write a poem about C programming."
```

---

## Architectural & Mathematical Overview

For an in-depth mathematical breakdown (Leaky ReLU, Sigmoid, Binary Cross-Entropy, RMSNorm, RoPE, SwiGLU, Self-Attention, and Metal MSL kernels), please refer to [`docs/documentation.md`](docs/documentation.md).

---

## License
Distributed under the MIT License. See `LICENSE` for details.
