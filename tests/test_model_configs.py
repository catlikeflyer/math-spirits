import json
from pathlib import Path
import pytest
import yaml

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def test_models_directory_exists():
    assert MODELS_DIR.exists(), f"Models directory {MODELS_DIR} does not exist"
    model_folders = [p for p in MODELS_DIR.iterdir() if p.is_dir()]
    assert len(model_folders) >= 2, "Expected at least 2 models (math-ghost-1, math-spectre-1)"


@pytest.mark.parametrize("model_name", ["math-ghost-1", "math-spectre-1"])
def test_runtime_config(model_name):
    config_path = MODELS_DIR / model_name / "configs" / "runtime_config.json"
    assert config_path.exists(), f"Missing runtime_config.json for {model_name}"

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "id" in data
    assert data["id"] == model_name
    assert "name" in data
    assert "base_model" in data
    assert "quantization" in data
    assert "parameters" in data
    assert "context_window" in data
    assert isinstance(data["context_window"], int)
    assert data["context_window"] > 0


@pytest.mark.parametrize("model_name", ["math-ghost-1", "math-spectre-1"])
def test_lora_config(model_name):
    config_path = MODELS_DIR / model_name / "configs" / "lora_config.yaml"
    assert config_path.exists(), f"Missing lora_config.yaml for {model_name}"

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), "lora_config.yaml must parse as a valid YAML mapping"
    assert "model" in data or "model_id" in data or "base_model" in data or "lora_parameters" in data
