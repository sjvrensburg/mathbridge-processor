import json
from pathlib import Path
import typer
from rich.console import Console
from .processor import MathBridgeProcessor
from .config import ConfigManager
from .schemas import ProcessingConfig

app = typer.Typer()
console = Console()


@app.command()
def process(
    output: str = typer.Option("mathbridge_processed", help="Output directory"),
    config: str = typer.Option(None, help="Path to JSON config"),
    batch_size: int = typer.Option(None, help="Batch size"),
    max_records: int = typer.Option(None, help="Max records to process"),
    resume_from: int = typer.Option(0, help="Resume index"),
    sre_domain: str = typer.Option("clearspeak", help="SRE domain (mathspeak|clearspeak)"),
    sre_locale: str = typer.Option("en", help="SRE locale"),
    latex2sre_path: str = typer.Option("./latex2sre", help="Path to latex2sre binary"),
    max_workers: int = typer.Option(None, help="Max parallel workers (default: auto-detect)"),
    verbose: bool = typer.Option(False, help="Verbose progress"),
):
    """
    Run the MathBridge processing pipeline.

    - Retains all original dataset columns, including 'spoken_English'
    - Adds one new column: 'sre_spoken_text'
    """
    if config:
        cfg = ConfigManager.from_json(config)
        # allow CLI overrides for output and others if provided
        cfg.output_path = output or cfg.output_path
        if batch_size is not None:
            cfg.batch_size = batch_size
        if max_records is not None:
            cfg.max_records = max_records
        if resume_from is not None:
            cfg.resume_from = resume_from
        if sre_domain:
            cfg.sre_domain = cfg.sre_domain.__class__(sre_domain)
        if sre_locale:
            cfg.sre_locale = sre_locale
        if latex2sre_path:
            cfg.latex2sre_path = latex2sre_path
        if max_workers is not None:
            cfg.max_workers = max_workers
    else:
        # build from CLI/env with defaults
        env_cfg = ConfigManager.from_env()
        cfg = ProcessingConfig(
            output_path=output or env_cfg.output_path,
            batch_size=batch_size or env_cfg.batch_size,
            max_records=max_records if max_records is not None else env_cfg.max_records,
            resume_from=resume_from or env_cfg.resume_from,
            sre_domain=env_cfg.sre_domain.__class__(sre_domain),
            sre_locale=sre_locale or env_cfg.sre_locale,
            latex2sre_path=latex2sre_path or env_cfg.latex2sre_path,
            max_workers=max_workers if max_workers is not None else env_cfg.max_workers,
        )

    processor = MathBridgeProcessor(cfg, verbose=verbose)
    result = processor.process_dataset()
    console.print("Processing complete.")
    console.print(json.dumps(result.to_dict(), indent=2))


@app.command()
def create_config(output: str = "config.json"):
    """Create config template."""
    ConfigManager.create_template(output)
    console.print(f"Wrote template to {output}")


@app.command()
def agent_info():
    """Show AI agent usage instructions."""
    console.print(
        """
AI Agent Usage:
- Use ConfigManager.create_template() to create a JSON config file
- Environment variables (override defaults):
  MB_SRE_DOMAIN, MB_SRE_LOCALE, MB_BATCH_SIZE, MB_MAX_RECORDS, MB_RESUME_FROM,
  MB_OUTPUT_PATH, MB_LATEX2SRE_PATH, MB_MAX_WORKERS
- Run: mathbridge-process process --config config.json
- Use --max-workers N to control parallel processing (default: auto-detect)
        """
    )


if __name__ == "__main__":
    app()
