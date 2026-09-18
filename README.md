# ✨ Math Spirits

> **Lightweight, open mathematical reasoning Small Language Models (SLMs) paired with an interactive chain-of-thought web engine.**

Math Spirits is a series of fine-tuned small language models trained specifically on mathematical reasoning and deduction tasks. The ultimate goal of this project is to provide high-quality educational AI models that run efficiently on common personal electronic devices—democratizing mathematical learning everywhere, anytime.

---

## 🧭 Philosophy

> *"I started math spirits in hopes to democratize fair AI usage in education. So much potential, that shouldn't be locked behind paywalls or high-end hardware."*

---

## ⚡ Key Features

- **On-Device Fine-Tuned Models**: Starting with **Spectre 1** (1.5B), fine-tuned on GSM8K and NuminaMath-CoT using LoRA for structured chain-of-thought deduction.
- **Chain-of-Thought Scratchpad**: Real-time parser intercepts `<thought>...</thought>` reasoning streams and presents them in an animated, collapsible accordion titled *"Spirits' Reasoning Scratchpad"*.
- **KaTeX Mathematical Typesetting**: Full rendering of mathematical notations, fractions, square roots, and boxed LaTeX results (`$...$`, `$$...$$`, `\boxed{...}`).
- **Dual-Mode Inference**:
  - **Backend Mode**: Connects to the local FastAPI server powered by Apple Silicon MLX (`mlx-lm`) with hot-swappable model management and Server-Sent Events (SSE).
  - **WebGPU / Offline Mode**: Client-side inference powered by `@mlc-ai/web-llm`, running quantized weights directly in-browser with zero server installation.
- **Live Inference Telemetry**: Real-time generation speed (tokens/sec), time-to-first-token (TTFT), token count, and Apple Metal VRAM footprint.
- **Transcript Export**: One-click **Export to Markdown** button to download full math sessions, formulas, and reasoning chains.

---

## 📂 Project Architecture

```text
math-spirits/
├── data/                                    # Centralized shared dataset
│   ├── train.jsonl                          # GSM8K + NuminaMath-CoT blend
│   ├── valid.jsonl
│   └── test.jsonl
├── scripts/                                 # Shared pipeline tooling
│   └── prepare_data.py                      # Data curation & normalization pipeline
├── models/                                  # Pure versioned model artifacts
│   └── math-spectre-1/
│       ├── configs/
│       │   ├── lora_config.yaml             # LoRA training & eval hyperparameters
│       │   └── runtime_config.json          # Server runtime metadata & specification
│       ├── adapters/
│       │   ├── adapter_config.json
│       │   └── adapters.safetensors         # Fine-tuned LoRA adapter weights
│       └── exports/
│           └── README.md                    # Quantization & GGUF export target info
├── server/                                  # FastAPI backend
│   ├── app.py                               # REST & SSE streaming server
│   ├── engine.py                            # Dynamic MLX / GGUF model manager
│   └── requirements.txt
└── web/                                     # Reactive UI (Vite + React + Tailwind)
    ├── index.html                           # KaTeX & typography
    ├── package.json                         # WebLLM, KaTeX, Lucide, Confetti
    └── src/
        ├── App.tsx                          # App root & dual-mode state
        ├── components/
        │   ├── TopBar.tsx                   # Model picker, badges, & mode toggle
        │   ├── PresetPills.tsx              # Quick math problem testing pills
        │   ├── ReasoningScratchpad.tsx      # Collapsible accordion for thoughts
        │   ├── FinalAnswerCard.tsx          # Highlighted final answer with copy button
        │   ├── TelemetryBadge.tsx           # Live tokens/sec, TTFT, and RAM metrics
        │   ├── MathRenderer.tsx             # KaTeX LaTeX renderer
        │   └── ChatCanvas.tsx               # Chat stream with math shortcuts
        └── services/
            ├── api.ts                       # FastAPI SSE streaming client
            └── webllm.ts                    # In-browser WebGPU inference service
```

---

## 🚀 Quick Start

### 1. Prerequisites

- macOS with Apple Silicon (recommended for MLX acceleration) or Linux/Windows.
- Python 3.10+
- Node.js 18+ and npm

### 2. Prepare the Dataset

Curate the blended GSM8K + NuminaMath-CoT dataset with `<thought>` tags and final answer formatting:

```bash
python3 scripts/prepare_data.py
```

### 3. Training & Evaluation with MLX

Fine-tune or evaluate the model using Apple Silicon MLX:

```bash
# Evaluate existing adapters
python3 -m mlx_lm lora --config models/math-spectre-1/configs/lora_config.yaml --test

# Train with LoRA
python3 -m mlx_lm lora --config models/math-spectre-1/configs/lora_config.yaml --train
```

### 4. Run the Inference Server

Install Python requirements and launch the FastAPI server:

```bash
pip install -r server/requirements.txt
python3 server/app.py
```
*The server will start on `http://localhost:8000` with automatic model discovery and health checks.*

### 5. Launch the Web Interface

Install frontend dependencies and start the Vite dev server:

```bash
cd web
npm install
npm run dev
```
*Open **`http://localhost:5173`** in your browser to interact with Math Spirits!*

---

## 🔮 Model Registry

| Model ID | Base Architecture | Parameters | Quantization | Adapter Format | Context Window |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`math-ghost-1`** | Qwen/Qwen2.5-0.5B-Instruct | 0.5B | Q4_K_M | MLX LoRA (`safetensors`) | 4096 |
| **`math-spectre-1`** | Qwen/Qwen2.5-Math-1.5B | 1.5B | Q4_K_M | MLX LoRA (`safetensors`) | 4096 |

---

## 📜 License

MIT License. Dedicated to open, democratized mathematical education.
