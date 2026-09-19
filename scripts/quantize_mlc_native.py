#!/usr/bin/env python3
"""
scripts/quantize_mlc_native.py
Native Python/PyTorch MLC q4f16_1 weight quantizer for Qwen2 / Qwen2.5 models.
Quantizes fused fine-tuned Hugging Face weights directly into WebGPU MLC shards.

No TVM compilation or JIT needed — runs 100% natively on macOS / Linux.
"""

import json
import shutil
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file


def pack_int4_weights(weight_tensor: torch.Tensor, group_size: int = 32):
    """
    Quantizes a 2D weight tensor (N, K) using MLC q4f16_1 symmetric group quantization.
    Returns:
        packed_qweight: np.ndarray of dtype uint32 with shape (N, K // 8)
        scales: np.ndarray of dtype float16 with shape (N, K // group_size)
    """
    w = weight_tensor.float()
    N, K = w.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    num_groups = K // group_size

    # Reshape into groups: (N, num_groups, group_size)
    w_grouped = w.view(N, num_groups, group_size)

    # Compute scales: max(abs(group)) / 7.0
    max_abs = torch.max(torch.abs(w_grouped), dim=2, keepdim=True).values
    max_abs = torch.clamp(max_abs, min=1e-4)
    scale = max_abs / 7.0  # (N, num_groups, 1)

    # Quantize: clamp(round(w / scale + 7), 0, 15)
    q = torch.clamp(torch.round(w_grouped / scale + 7.0), 0, 15).to(torch.uint8)
    q = q.view(N, K)

    # Pack 8 4-bit integers into each uint32
    # uint32 = q0 | (q1 << 4) | (q2 << 8) | (q3 << 12) | (q4 << 16) | (q5 << 20) | (q6 << 24) | (q7 << 28)
    q_np = q.cpu().numpy().astype(np.uint32)
    q_reshaped = q_np.reshape(N, K // 8, 8)

    shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    packed_qweight = np.bitwise_or.reduce(q_reshaped << shifts, axis=2).astype(np.uint32)

    scales_np = scale.squeeze(2).cpu().to(torch.float16).numpy().astype(np.float16)

    return packed_qweight, scales_np


def load_model_weights(fused_dir: Path) -> dict:
    """Loads all safetensors into a dictionary of PyTorch tensors."""
    weights = {}
    safetensor_files = sorted(fused_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors found in {fused_dir}")
    for sf in safetensor_files:
        tensors = load_file(str(sf))
        weights.update(tensors)
    return weights


def convert_model(model_name: str, fused_dir: Path, output_dir: Path):
    print(f"\n{'=' * 60}")
    print(f"Quantizing {model_name}")
    print(f"  Input  : {fused_dir}")
    print(f"  Output : {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_weights = load_model_weights(fused_dir)
    print(f"  Loaded {len(raw_weights)} raw weight tensors")

    # Read config.json
    with open(fused_dir / "config.json") as f:
        config = json.load(f)

    num_layers = config.get("num_hidden_layers", 28)
    tie_word_embeddings = config.get("tie_word_embeddings", False)

    mlc_tensors = {}

    # 1. Embeddings
    print("  Quantizing embed_tokens...")
    embed_w = raw_weights["model.embed_tokens.weight"]
    q_w, q_s = pack_int4_weights(embed_w)
    mlc_tensors["model.embed_tokens.q_weight"] = q_w
    mlc_tensors["model.embed_tokens.q_scale"] = q_s

    # 2. Layers
    for i in range(num_layers):
        pfx = f"model.layers.{i}"
        print(f"  Quantizing layer {i + 1}/{num_layers}...", end="\r", flush=True)

        # Input LayerNorm
        mlc_tensors[f"{pfx}.input_layernorm.weight"] = (
            raw_weights[f"{pfx}.input_layernorm.weight"].cpu().to(torch.float16).numpy()
        )

        # Attention Q, K, V -> c_attn
        q_proj = raw_weights[f"{pfx}.self_attn.q_proj.weight"]
        k_proj = raw_weights[f"{pfx}.self_attn.k_proj.weight"]
        v_proj = raw_weights[f"{pfx}.self_attn.v_proj.weight"]
        c_attn_w = torch.cat([q_proj, k_proj, v_proj], dim=0)

        q_w, q_s = pack_int4_weights(c_attn_w)
        mlc_tensors[f"{pfx}.self_attn.c_attn.q_weight"] = q_w
        mlc_tensors[f"{pfx}.self_attn.c_attn.q_scale"] = q_s

        if f"{pfx}.self_attn.q_proj.bias" in raw_weights:
            q_b = raw_weights[f"{pfx}.self_attn.q_proj.bias"]
            k_b = raw_weights[f"{pfx}.self_attn.k_proj.bias"]
            v_b = raw_weights[f"{pfx}.self_attn.v_proj.bias"]
            c_attn_b = torch.cat([q_b, k_b, v_b], dim=0).cpu().to(torch.float16).numpy()
            mlc_tensors[f"{pfx}.self_attn.c_attn.bias"] = c_attn_b

        # Attention O
        o_proj = raw_weights[f"{pfx}.self_attn.o_proj.weight"]
        q_w, q_s = pack_int4_weights(o_proj)
        mlc_tensors[f"{pfx}.self_attn.o_proj.q_weight"] = q_w
        mlc_tensors[f"{pfx}.self_attn.o_proj.q_scale"] = q_s

        # Post Attention LayerNorm
        mlc_tensors[f"{pfx}.post_attention_layernorm.weight"] = (
            raw_weights[f"{pfx}.post_attention_layernorm.weight"].cpu().to(torch.float16).numpy()
        )

        # MLP Gate + Up -> gate_up_proj
        gate_proj = raw_weights[f"{pfx}.mlp.gate_proj.weight"]
        up_proj = raw_weights[f"{pfx}.mlp.up_proj.weight"]
        gate_up_w = torch.cat([gate_proj, up_proj], dim=0)
        q_w, q_s = pack_int4_weights(gate_up_w)
        mlc_tensors[f"{pfx}.mlp.gate_up_proj.q_weight"] = q_w
        mlc_tensors[f"{pfx}.mlp.gate_up_proj.q_scale"] = q_s

        # MLP Down
        down_proj = raw_weights[f"{pfx}.mlp.down_proj.weight"]
        q_w, q_s = pack_int4_weights(down_proj)
        mlc_tensors[f"{pfx}.mlp.down_proj.q_weight"] = q_w
        mlc_tensors[f"{pfx}.mlp.down_proj.q_scale"] = q_s

    print(f"\n  Quantized all {num_layers} layers successfully.")

    # 3. Final Norm & LM Head
    mlc_tensors["model.norm.weight"] = (
        raw_weights["model.norm.weight"].cpu().to(torch.float16).numpy()
    )

    if not tie_word_embeddings and "lm_head.weight" in raw_weights:
        lm_head_w = raw_weights["lm_head.weight"]
        q_w, q_s = pack_int4_weights(lm_head_w)
        mlc_tensors["lm_head.q_weight"] = q_w
        mlc_tensors["lm_head.q_scale"] = q_s

    # 4. Write binary shards and ndarray-cache.json
    print("  Writing binary shards (max 128MB per shard)...")
    MAX_SHARD_BYTES = 128 * 1024 * 1024  # 128 MB

    shard_idx = 0
    current_shard_bytes = bytearray()
    shard_records = []
    all_shard_meta = []

    for name, arr in sorted(mlc_tensors.items()):
        raw_bytes = arr.tobytes()
        nbytes = len(raw_bytes)

        if len(current_shard_bytes) + nbytes > MAX_SHARD_BYTES and len(current_shard_bytes) > 0:
            shard_name = f"params_shard_{shard_idx}.bin"
            with open(output_dir / shard_name, "wb") as sf:
                sf.write(current_shard_bytes)
            all_shard_meta.append({
                "dataPath": shard_name,
                "format": "raw-shard",
                "nbytes": len(current_shard_bytes),
                "records": shard_records,
            })
            shard_idx += 1
            current_shard_bytes = bytearray()
            shard_records = []

        byte_offset = len(current_shard_bytes)
        current_shard_bytes.extend(raw_bytes)

        dtype_str = "uint32" if arr.dtype == np.uint32 else "float16"
        shard_records.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": dtype_str,
            "format": "f32-to-bf16",
            "nbytes": nbytes,
            "byteOffset": byte_offset,
        })

    # Flush last shard
    if len(current_shard_bytes) > 0:
        shard_name = f"params_shard_{shard_idx}.bin"
        with open(output_dir / shard_name, "wb") as sf:
            sf.write(current_shard_bytes)
        all_shard_meta.append({
            "dataPath": shard_name,
            "format": "raw-shard",
            "nbytes": len(current_shard_bytes),
            "records": shard_records,
        })

    # Write ndarray-cache.json
    manifest = {"records": all_shard_meta}
    with open(output_dir / "ndarray-cache.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # 5. Generate mlc-chat-config.json
    mlc_chat_config = {
        "model_type": "qwen2",
        "quantization": "q4f16_1",
        "model_config": config,
        "vocab_size": config.get("vocab_size", 151936),
        "context_window_size": config.get("max_position_embeddings", 32768),
        "prefill_chunk_size": 2048,
        "sliding_window_size": -1,
        "attention_sink_size": -1,
        "tensor_parallel_shards": 1,
        "conv_template": "chatml",
        "temperature": 0.6,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.1,
        "top_p": 0.95,
        "tokenizer_files": ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
    }
    with open(output_dir / "mlc-chat-config.json", "w") as f:
        json.dump(mlc_chat_config, f, indent=2)

    # 6. Copy tokenizer files
    for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json"]:
        src = fused_dir / fn
        if src.exists():
            shutil.copy2(src, output_dir / fn)

    print(f"  [DONE] {model_name} quantized successfully -> {output_dir}")
    print(f"  Created {shard_idx + 1} shard files, total size: {sum(s['nbytes'] for s in all_shard_meta) / (1024*1024):.1f} MB")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    dist_root = repo_root / "dist" / "mlc"

    ghost_fused = repo_root / "models" / "math-ghost-1" / "fused"
    spectre_fused = repo_root / "models" / "math-spectre-1" / "fused"

    if ghost_fused.exists():
        convert_model("math-ghost-1", ghost_fused, dist_root / "math-ghost-1-q4f16_1-MLC")
    else:
        print(f"[SKIP] {ghost_fused} does not exist.")

    if spectre_fused.exists():
        convert_model("math-spectre-1", spectre_fused, dist_root / "math-spectre-1-q4f16_1-MLC")
    else:
        print(f"[SKIP] {spectre_fused} does not exist.")


if __name__ == "__main__":
    main()
