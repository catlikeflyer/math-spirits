import os
import gc
import json
import time
import queue
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Generator, Optional, List

import psutil
import mlx.core as mx
import mlx_lm
from mlx_lm.sample_utils import make_sampler

try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    llama_cpp = None
    LLAMA_CPP_AVAILABLE = False

logger = logging.getLogger("math_spirits.engine")
logging.basicConfig(level=logging.INFO)

DEFAULT_SYSTEM_PROMPT = (
    "You are Spirits, an expert math tutor. Always think through the solution "
    "step-by-step inside <thought> tags before giving the final answer."
)


class ModelEngine:
    """
    Thread-safe model engine for MLX and llama.cpp.
    Uses a dedicated inference worker thread so that all MLX arrays and Metal
    streams reside consistently within the same thread context.
    """
    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = Path(workspace_root or Path(__file__).resolve().parent.parent)
        self.models_dir = self.workspace_root / "models"
        
        self.registered_models: Dict[str, Dict[str, Any]] = {}
        self.active_model_id: Optional[str] = None
        self.active_model_type: Optional[str] = None
        
        # Model handles (managed exclusively on the worker thread)
        self._mlx_model = None
        self._mlx_tokenizer = None
        self._llama_model = None
        
        # Task queue and dedicated worker thread
        self._task_queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="MLX-Inference-Worker")
        self._worker_thread.start()
        
        self.discover_models()
        
        # Auto-select math-spectre-1 if present
        if "math-spectre-1" in self.registered_models:
            self.select_model("math-spectre-1")
        elif self.registered_models:
            first_model = next(iter(self.registered_models.keys()))
            self.select_model(first_model)

    def _worker_loop(self):
        """Dedicated execution loop for all MLX / llama.cpp memory operations."""
        logger.info("Inference worker thread started.")
        while True:
            task = self._task_queue.get()
            if task is None:
                break
                
            task_type, payload, reply_queue = task
            try:
                if task_type == "select_model":
                    model_id = payload["model_id"]
                    meta = self._load_model_on_worker(model_id)
                    reply_queue.put({"success": True, "meta": meta})
                    
                elif task_type == "generate":
                    self._generate_on_worker(payload, reply_queue)
                    
            except Exception as exc:
                logger.error(f"Error in inference worker task '{task_type}': {exc}", exc_info=True)
                reply_queue.put({"type": "error", "error": str(exc)})
                reply_queue.put(None)
            finally:
                self._task_queue.task_done()

    def _load_model_on_worker(self, model_id: str) -> Dict[str, Any]:
        """Loads a model inside the worker thread."""
        if model_id not in self.registered_models:
            raise ValueError(f"Model '{model_id}' not found in registered models.")

        if self.active_model_id == model_id and (self._mlx_model is not None or self._llama_model is not None):
            logger.info(f"Model '{model_id}' is already loaded on worker thread.")
            return self.registered_models[model_id]

        # Unload existing
        if self._mlx_model is not None or self._llama_model is not None:
            logger.info(f"Unloading current model ({self.active_model_id}) on worker...")
            self._mlx_model = None
            self._mlx_tokenizer = None
            self._llama_model = None
            gc.collect()
            try:
                mx.metal.clear_cache()
            except Exception:
                pass

        model_meta = self.registered_models[model_id]
        mtype = model_meta.get("type", "mlx").lower()

        logger.info(f"Worker loading model '{model_id}' (type: {mtype})...")
        t0 = time.perf_counter()

        if mtype == "mlx":
            base_model = model_meta.get("base_model", "Qwen/Qwen2.5-Math-1.5B")
            adapter_path = model_meta.get("adapter_path")
            
            if adapter_path and not Path(adapter_path).exists():
                logger.warning(f"Adapter path {adapter_path} not found. Attempting fallback...")
                root_path = Path(model_meta.get("root_path", ""))
                candidates = list(root_path.glob("**/adapters.safetensors"))
                if candidates:
                    adapter_path = str(candidates[0].parent)
                    model_meta["adapter_path"] = adapter_path
                    logger.info(f"Found adapter at: {adapter_path}")
                else:
                    adapter_path = None

            logger.info(f"Loading MLX base model '{base_model}' with adapter '{adapter_path}'...")
            if adapter_path:
                self._mlx_model, self._mlx_tokenizer = mlx_lm.load(base_model, adapter_path=adapter_path)
            else:
                self._mlx_model, self._mlx_tokenizer = mlx_lm.load(base_model)
            self.active_model_type = "mlx"

            # Register ChatML and standard end tokens into tokenizer eos_token_ids
            for st in ["<|im_end|>", "<|endoftext|>", "</s>"]:
                try:
                    tid = self._mlx_tokenizer.convert_tokens_to_ids(st)
                    if tid is not None and isinstance(tid, int) and tid > 0:
                        if hasattr(self._mlx_tokenizer, "eos_token_ids"):
                            if isinstance(self._mlx_tokenizer.eos_token_ids, set):
                                self._mlx_tokenizer.eos_token_ids.add(tid)
                            elif isinstance(self._mlx_tokenizer.eos_token_ids, list):
                                self._mlx_tokenizer.eos_token_ids.append(tid)
                except Exception:
                    pass

        elif mtype == "gguf":
            if not LLAMA_CPP_AVAILABLE:
                raise RuntimeError("llama-cpp-python is not installed.")
            gguf_path = model_meta.get("export_gguf_path") or model_meta.get("gguf_path")
            if not gguf_path or not Path(gguf_path).exists():
                raise FileNotFoundError(f"GGUF binary not found at '{gguf_path}'")
            n_ctx = model_meta.get("context_window", 4096)
            self._llama_model = llama_cpp.Llama(model_path=str(gguf_path), n_ctx=n_ctx, n_gpu_layers=-1)
            self.active_model_type = "gguf"
        else:
            raise ValueError(f"Unsupported model type: {mtype}")

        self.active_model_id = model_id
        load_time = time.perf_counter() - t0
        logger.info(f"Worker successfully loaded '{model_id}' in {load_time:.2f}s")
        return model_meta

    def _generate_on_worker(self, payload: Dict[str, Any], out_queue: queue.Queue):
        """Runs streaming generation on the worker thread."""
        prompt = payload["prompt"]
        temperature = payload.get("temperature", 0.7)
        max_tokens = payload.get("max_tokens", 1024)

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        generated_tokens = 0
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "</s>"]
        stop_token_ids = set()
        if self._mlx_tokenizer:
            for st in stop_tokens:
                try:
                    tid = self._mlx_tokenizer.convert_tokens_to_ids(st)
                    if tid is not None and isinstance(tid, int) and tid > 0:
                        stop_token_ids.add(tid)
                except Exception:
                    pass

        accumulated_text = ""
        if self.active_model_type == "mlx":
            sampler = make_sampler(temp=max(0.0, float(temperature)))
            for resp in mlx_lm.stream_generate(
                self._mlx_model,
                self._mlx_tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                if resp.token in stop_token_ids:
                    break

                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now

                text_chunk = resp.text
                should_stop = False
                # Check for stop tokens inside chunk
                for st in stop_tokens:
                    if st in text_chunk:
                        text_chunk = text_chunk.split(st)[0]
                        should_stop = True
                        break

                # Stop if hallucinated OCR tags or image links appear
                for bad_tag in ["</latex>", "![", "<|im_end|>", "<|endoftext|>", "alignSelf"]:
                    if bad_tag in text_chunk:
                        text_chunk = text_chunk.split(bad_tag)[0]
                        should_stop = True
                        break

                # Check if we have encountered "Final Answer:"
                accumulated_with_chunk = accumulated_text + text_chunk
                fa_lower = accumulated_with_chunk.lower()
                first_fa = fa_lower.find("final answer:")
                if first_fa != -1:
                    after_fa = accumulated_with_chunk[first_fa + 13:]
                    after_fa_stripped = after_fa.lstrip()
                    # If we already have some answer text, stop at the first newline or scraper delimiter
                    if after_fa_stripped:
                        for delim in ["\n", "</latex>", "<|im_end|>", "<|endoftext|>", "![", "【"]:
                            if delim in after_fa_stripped:
                                # Truncate text_chunk so delimiter and trailing junk are not emitted
                                if delim in text_chunk:
                                    text_chunk = text_chunk.split(delim)[0]
                                should_stop = True
                                break

                    # Also stop if another "final answer:" loop starts
                    if fa_lower.find("final answer:", first_fa + 13) != -1:
                        should_stop = True

                accumulated_text += text_chunk

                if text_chunk:
                    generated_tokens += 1
                    out_queue.put({
                        "type": "token",
                        "content": text_chunk,
                        "token_count": generated_tokens,
                    })

                if should_stop or resp.finish_reason in ["stop", "length"]:
                    break

        elif self.active_model_type == "gguf":
            messages = payload.get("messages", [])
            for chunk in self._llama_model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            ):
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now

                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    generated_tokens += 1
                    out_queue.put({
                        "type": "token",
                        "content": content,
                        "token_count": generated_tokens,
                    })

        end_time = time.perf_counter()
        total_duration = end_time - start_time
        ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
        gen_duration = end_time - first_token_time if first_token_time else total_duration
        tokens_per_second = (generated_tokens / gen_duration) if gen_duration > 0 and generated_tokens > 0 else 0.0

        try:
            active_mem_gb = mx.get_active_memory() / (1024 ** 3)
            peak_mem_gb = mx.get_peak_memory() / (1024 ** 3)
        except Exception:
            active_mem_gb = 0.0
            peak_mem_gb = 0.0

        telemetry = {
            "tokens_per_second": round(tokens_per_second, 2),
            "time_to_first_token_ms": round(ttft_ms, 2),
            "total_tokens": generated_tokens,
            "total_duration_s": round(total_duration, 2),
            "active_memory_gb": round(active_mem_gb, 2),
            "peak_memory_gb": round(peak_mem_gb, 2),
            "model_id": self.active_model_id,
        }

        out_queue.put({
            "type": "done",
            "telemetry": telemetry,
        })
        out_queue.put(None)  # Sentinel to signify completion

    def discover_models(self) -> Dict[str, Dict[str, Any]]:
        """Scans models/ and workspace directories to discover available models."""
        self.registered_models.clear()
        
        # Check standard models/ directory
        if self.models_dir.exists() and self.models_dir.is_dir():
            for entry in self.models_dir.iterdir():
                if entry.is_dir():
                    config_path = entry / "configs" / "runtime_config.json"
                    if config_path.exists():
                        try:
                            with open(config_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            model_id = data.get("id", entry.name)
                            data["root_path"] = str(entry)
                            self.registered_models[model_id] = data
                            logger.info(f"Discovered model from config: {model_id}")
                        except Exception as e:
                            logger.error(f"Failed to read config from {config_path}: {e}")

        # Fallback check for models/math-spectre-1 directory
        if "math-spectre-1" not in self.registered_models:
            alt_path = self.models_dir / "math-spectre-1"
            if alt_path.exists():
                adapter_dir = alt_path / "adapters"
                self.registered_models["math-spectre-1"] = {
                    "id": "math-spectre-1",
                    "name": "Math Spectre 1",
                    "display_name": "Spectre 1 (Math Spirit)",
                    "base_model": "Qwen/Qwen2.5-Math-1.5B",
                    "adapter_path": str(adapter_dir),
                    "quantization": "Q4_K_M",
                    "parameters": "1.5B",
                    "context_window": 4096,
                    "type": "mlx",
                    "system_prompt": DEFAULT_SYSTEM_PROMPT,
                    "description": "Fine-tuned Qwen2.5-Math-1.5B with step-by-step chain-of-thought reasoning scratchpad on GSM8K and NuminaMath.",
                    "root_path": str(alt_path)
                }
                logger.info("Registered math-spectre-1 from default model path.")

        # Ensure adapter path is resolved
        for mid, meta in self.registered_models.items():
            if "adapter_path" in meta and meta["adapter_path"]:
                ad_path = Path(meta["adapter_path"])
                if not ad_path.is_absolute():
                    meta["adapter_path"] = str((self.workspace_root / ad_path).resolve())

        return self.registered_models

    def select_model(self, model_id: str) -> Dict[str, Any]:
        """Hot-swaps the active model by dispatching to the worker thread."""
        if model_id not in self.registered_models:
            self.discover_models()
            if model_id not in self.registered_models:
                raise ValueError(f"Model '{model_id}' not found in registered models.")

        reply_q = queue.Queue()
        self._task_queue.put(("select_model", {"model_id": model_id}, reply_q))
        res = reply_q.get()
        if not res.get("success"):
            raise RuntimeError(res.get("error", "Failed to load model"))
        return res["meta"]

    def format_chatml_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Formats multi-turn messages into ChatML syntax."""
        meta = self.registered_models.get(self.active_model_id, {})
        effective_system = system_prompt or meta.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        
        has_system = any(m.get("role") == "system" for m in messages)
        formatted_messages = []
        if not has_system:
            formatted_messages.append({"role": "system", "content": effective_system})
        formatted_messages.extend(messages)
        
        prompt_parts = []
        for msg in formatted_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        prompt_parts.append("<|im_start|>assistant\n")
        return "".join(prompt_parts)

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Dispatches generation to the worker thread and yields SSE-compatible token events.
        """
        target_model = model_id or self.active_model_id or "math-spectre-1"
        if self.active_model_id != target_model:
            self.select_model(target_model)

        prompt = self.format_chatml_prompt(messages, system_prompt=system_prompt)
        payload = {
            "prompt": prompt,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        out_queue = queue.Queue()
        self._task_queue.put(("generate", payload, out_queue))

        while True:
            item = out_queue.get()
            if item is None:
                break
            if isinstance(item, dict) and item.get("type") == "error":
                raise RuntimeError(item.get("error", "Generation error"))
            yield item

    def get_system_telemetry(self) -> Dict[str, Any]:
        """Returns unified RAM and Apple Silicon Metal GPU memory metrics."""
        vm = psutil.virtual_memory()
        
        try:
            active_metal_gb = mx.get_active_memory() / (1024 ** 3)
            peak_metal_gb = mx.get_peak_memory() / (1024 ** 3)
        except Exception:
            active_metal_gb = 0.0
            peak_metal_gb = 0.0

        active_meta = self.registered_models.get(self.active_model_id, {})

        return {
            "active_model_id": self.active_model_id,
            "active_model_name": active_meta.get("display_name", active_meta.get("name", "None")),
            "quantization": active_meta.get("quantization", "Unknown"),
            "parameters": active_meta.get("parameters", "Unknown"),
            "ram_used_gb": round(vm.used / (1024 ** 3), 2),
            "ram_total_gb": round(vm.total / (1024 ** 3), 2),
            "ram_percent": vm.percent,
            "metal_active_gb": round(active_metal_gb, 2),
            "metal_peak_gb": round(peak_metal_gb, 2),
            "is_loaded": (self.active_model_id is not None),
        }
