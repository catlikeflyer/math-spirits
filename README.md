# ✨ Math Spirits

> **Lightweight, open mathematical reasoning Small Language Models (SLMs) paired with an interactive chain-of-thought web engine.**

Math Spirits is a series of fine-tuned small language models trained specifically on mathematical reasoning and deduction tasks. The ultimate goal of this project is to provide high-quality educational AI models that run efficiently on common personal electronic devices—democratizing mathematical learning everywhere, anytime without high-end GPUs or subscription paywalls.

---

## 🧭 Philosophy

> *"I started math spirits in hopes to democratize fair AI usage in education. So much potential, that shouldn't be locked behind paywalls or high-end hardware."*

---

## ⚡ Key Features

- **On-Device Fine-Tuned Models**:
  - **Math Ghost 1 (0.5B)**: Ultra-lightweight fine-tuned spirit designed for sub-second mathematical deduction on low-spec laptops and mobile devices.
  - **Math Spectre 1 (1.5B)**: Deep mathematical tutor spirit fine-tuned on GSM8K and NuminaMath-CoT for multi-step derivations and proofs.
- **Client-Side WebGPU Mode**: Run fine-tuned model weights directly inside any modern browser using `@mlc-ai/web-llm` with zero backend server dependencies.
- **Dual-Mode Architecture**:
  - **WebGPU / Offline Mode**: In-browser client inference powered by WebGPU and MLC q4f16_1 quantization. Weights stream directly from Hugging Face into browser GPU memory.
  - **Backend Mode**: Connects to the local FastAPI + Apple Silicon MLX (`mlx-lm`) server with hot-swappable model switching and Server-Sent Events (SSE).
- **Chain-of-Thought Scratchpad**: Real-time parser intercepts `<thought>...</thought>` reasoning streams and presents them in an interactive, collapsible accordion.
- **KaTeX Mathematical Typesetting**: Full LaTeX rendering of mathematical notation, fractions, square roots, and boxed answers (`$...$`, `$$...$$`, `\boxed{...}`).
- **Live Inference Telemetry**: Real-time generation speed (tokens/sec), time-to-first-token (TTFT), token counts, and Apple Metal VRAM footprint.
- **Session Transcript Export**: One-click Markdown export to save complete math sessions, derivations, and telemetry.

---

## 🔮 Model Registry & Hugging Face

| Model ID | Display Name | Parameters | Architecture | Quantization | Hugging Face Repo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`math-ghost-1`** | Ghost 1 (Ultra-Light) | **0.5B** | Qwen2.5-0.5B | Q4_K_M / q4f16_1 | [`catlikeflyer/math-ghost-1-q4f16_1-MLC`](https://huggingface.co/catlikeflyer/math-ghost-1-q4f16_1-MLC) |
| **`math-spectre-1`** | Spectre 1 (Math Spirit) | **1.5B** | Qwen2.5-Math-1.5B | Q4_K_M / q4f16_1 | [`catlikeflyer/math-spectre-1-q4f16_1-MLC`](https://huggingface.co/catlikeflyer/math-spectre-1-q4f16_1-MLC) |

---

## 📂 Project Architecture

```text
math-spirits/
├── data/                                    # Centralized shared dataset
│   ├── train.jsonl                          # GSM8K + NuminaMath-CoT blend
│   ├── valid.jsonl
│   └── test.jsonl
├── scripts/                                 # Pipelines & automation
│   ├── prepare_data.py                      # Data curation & normalization pipeline
│   ├── quantize_mlc_native.py               # MLC q4f16_1 WebGPU quantization
│   ├── compile_mlc.sh                       # Batch MLC quantization runner
│   └── upload_hf.sh                         # HuggingFace Hub uploader
├── models/                                  # Versioned model artifacts
│   ├── math-ghost-1/                        # 0.5B Ultra-light spirit
│   │   ├── configs/
│   │   │   ├── lora_config.yaml             # LoRA hyperparameters
│   │   │   └── runtime_config.json          # Server metadata & runtime specs
│   │   └── adapters/                        # Fine-tuned LoRA weights
│   └── math-spectre-1/                      # 1.5B Chain-of-thought spirit
│       ├── configs/
│       │   ├── lora_config.yaml
│       │   └── runtime_config.json
│       └── adapters/
├── server/                                  # FastAPI backend
│   ├── app.py                               # REST & SSE streaming server
│   ├── engine.py                            # Dynamic MLX model manager
│   └── requirements.txt
└── web/                                     # Reactive UI (Vite + React + Tailwind)
    ├── package.json                         # WebLLM, KaTeX, Lucide
    └── src/
        ├── App.tsx                          # App root & dual-mode state
        ├── components/
        │   ├── TopBar.tsx                   # Model picker, badges, & mode toggle
        │   ├── PresetPills.tsx              # Quick math problem testing pills
        │   ├── ReasoningScratchpad.tsx      # Collapsible accordion for thoughts
        │   ├── FinalAnswerCard.tsx          # Highlighted final answer card
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

- macOS with Apple Silicon (recommended for local MLX acceleration) or any modern browser with WebGPU support (Chrome, Edge, Brave).
- Python 3.10+
- Node.js 18+ and npm

---

### 2. Standalone WebGPU Mode (Zero Backend)

You can run Math Spirits entirely inside your browser with no Python backend:

```bash
cd web
npm install
npm run dev
```

1. Open `http://localhost:5173`.
2. Toggle the engine to **WebGPU** mode in the top bar.
3. Select **Ghost 1 (0.5B)** or **Spectre 1 (1.5B)** from the dropdown.
4. The model will stream directly from Hugging Face and execute locally on your GPU.

---

### 3. Local MLX Backend Mode (Apple Silicon)

For ultra-low latency inference using Apple Silicon Metal (`mlx-lm`):

#### Install Backend Requirements
```bash
pip install -r server/requirements.txt
```

#### Launch the FastAPI Server
```bash
python3 server/app.py
```
*Server starts on `http://localhost:8000` with automated model discovery and health checks.*

#### Launch the Frontend
```bash
cd web
npm install
npm run dev
```
*In the top bar, keep **Backend** selected to route prompts to your local MLX engine.*

---

### 4. Training & Data Curation

#### Prepare Dataset
Curate the blended GSM8K + NuminaMath-CoT dataset with `<thought>` tags and boxed answers:
```bash
python3 scripts/prepare_data.py
```

#### Train / Evaluate LoRA Adapters with MLX
```bash
# Evaluate existing adapters
python3 -m mlx_lm lora --config models/math-spectre-1/configs/lora_config.yaml --test

# Train Spectre 1
python3 -m mlx_lm lora --config models/math-spectre-1/configs/lora_config.yaml --train
```

#### Export & Quantize for WebGPU
```bash
# Quantize fused models to MLC q4f16_1 format
bash scripts/compile_mlc.sh

# Upload to HuggingFace
HF_USERNAME=your_username bash scripts/upload_hf.sh
```

---

## 📜 License

MIT License. Dedicated to open, democratized mathematical education.
