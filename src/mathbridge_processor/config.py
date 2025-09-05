import json
import os
from typing import Optional

from .schemas import ProcessingConfig


class ConfigManager:
    @staticmethod
    def from_json(config_path: str) -> ProcessingConfig:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ProcessingConfig(**data)

    @staticmethod
    def from_env() -> ProcessingConfig:
        data = {
            "sre_domain": os.getenv("MB_SRE_DOMAIN"),
            "sre_locale": os.getenv("MB_SRE_LOCALE"),
            "batch_size": int(os.getenv("MB_BATCH_SIZE", "1000")),
            "max_records": int(os.getenv("MB_MAX_RECORDS")) if os.getenv("MB_MAX_RECORDS") else None,
            "resume_from": int(os.getenv("MB_RESUME_FROM", "0")),
            "output_path": os.getenv("MB_OUTPUT_PATH", "mathbridge_processed"),
            "latex2sre_path": os.getenv("MB_LATEX2SRE_PATH", "./latex2sre"),
        }
        # Drop None values to let pydantic defaults apply
        data = {k: v for k, v in data.items() if v is not None}
        return ProcessingConfig(**data)

    @staticmethod
    def create_template(output_path: str) -> None:
        cfg = ProcessingConfig().dict()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
