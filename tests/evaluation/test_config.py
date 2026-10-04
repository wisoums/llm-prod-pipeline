from pathlib import Path

import yaml


def test_base_config_has_required_sections():
    config_path = Path("configs/base.yaml")
    config = yaml.safe_load(config_path.read_text())
    assert {"model", "training", "tracking", "evaluation", "serving"} <= set(config)
    assert config["training"]["method"] == "qlora"
    assert config["serving"]["engine"] == "vllm"
