from server.engine import ModelEngine, DEFAULT_SYSTEM_PROMPT


def test_engine_discovery():
    engine = ModelEngine()
    models = engine.discover_models()
    assert isinstance(models, dict)
    assert "math-ghost-1" in models or "math-spectre-1" in models


def test_chatml_prompt_formatting():
    engine = ModelEngine()
    messages = [
        {"role": "user", "content": "What is 1 + 1?"}
    ]
    prompt = engine.format_chatml_prompt(messages)
    assert "<|im_start|>system" in prompt
    assert DEFAULT_SYSTEM_PROMPT in prompt
    assert "<|im_start|>user\nWhat is 1 + 1?<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_chatml_custom_system_prompt():
    engine = ModelEngine()
    custom_sys = "You are a geometry expert."
    messages = [
        {"role": "user", "content": "Calculate the area of a circle."}
    ]
    prompt = engine.format_chatml_prompt(messages, system_prompt=custom_sys)
    assert custom_sys in prompt
    assert DEFAULT_SYSTEM_PROMPT not in prompt


def test_system_telemetry():
    engine = ModelEngine()
    telemetry = engine.get_system_telemetry()
    assert isinstance(telemetry, dict)
    assert "ram_used_gb" in telemetry
    assert "ram_total_gb" in telemetry
    assert "ram_percent" in telemetry
    assert "metal_active_gb" in telemetry
    assert "metal_peak_gb" in telemetry
