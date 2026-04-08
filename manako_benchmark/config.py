"""Benchmark configuration — load from YAML or environment."""
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BenchConfig:
    """Full benchmark configuration."""
    # Dataset
    frames_dir: str = "data/frames"
    annotations_path: str = "data/annotations/annotations.json"

    # SN44
    sn44_weights: str | None = None
    sn44_hf_repo: str = "alfred8995/kane001"
    sn44_conf_threshold: float = 0.45

    # SAM3
    sam3_endpoint: str = ""
    sam3_api_key: str = ""

    # Roboflow
    roboflow_api_key: str = ""
    roboflow_model_id: str = "vehicles-q0x2v/1"
    roboflow_use_local: bool = False

    # Evaluation
    device: str = "cpu"
    models: list[str] = field(default_factory=lambda: ["sn44", "roboflow", "sam3"])

    # Output
    results_dir: str = "results"
    reports_dir: str = "reports"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "BenchConfig":
        """Load config from environment variables (MANAKO_* prefix)."""
        config = cls()
        for field_name in cls.__dataclass_fields__:
            env_key = f"MANAKO_{field_name.upper()}"
            val = os.environ.get(env_key)
            if val is not None:
                field_type = cls.__dataclass_fields__[field_name].type
                if field_type == "bool":
                    val = val.lower() in ("true", "1", "yes")
                elif "list" in str(field_type):
                    val = [v.strip() for v in val.split(",")]
                setattr(config, field_name, val)
        return config
