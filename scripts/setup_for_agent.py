#!/usr/bin/env python3
import json
import os
from pathlib import Path


def main():
    """Setup environment for AI agents."""
    # Create a default config template if not present
    cfg_path = Path("config.template.json")
    if not cfg_path.exists():
        template = {
            "sre_domain": "clearspeak",
            "sre_locale": "en",
            "batch_size": 1000,
            "max_records": None,
            "resume_from": 0,
            "output_path": "mathbridge_processed",
            "latex2sre_path": "./latex2sre",
        }
        cfg_path.write_text(json.dumps(template, indent=2))
        print(f"Wrote {cfg_path}")
    else:
        print("Template already exists")

    # Create outputs directory ignore file
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / ".gitkeep").write_text("")


if __name__ == "__main__":
    main()
