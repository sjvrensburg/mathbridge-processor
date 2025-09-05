import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from .schemas import ProcessingConfig, ProcessingResult, ProcessingStats
from .data_cleaner import MathBridgeCleaner, CleaningStats


logger = logging.getLogger(__name__)


class MathBridgeProcessor:
    def __init__(self, config: ProcessingConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.cleaner = MathBridgeCleaner()
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
        logger.debug("Initialized MathBridgeProcessor with config: %s", self.config)

    def _verify_tools(self):
        """Verify latex-validator and latex2sre are available."""
        if shutil.which("latex-validator") is None:
            logger.warning("latex-validator not found in PATH. Validation will be skipped.")
        
        # Try to find latex2sre in multiple locations
        latex2sre_paths = [
            self.config.latex2sre_path,
            shutil.which("latex2sre"),
            "/home/stefan/bin/latex2sre-linux-x64",
            "/usr/local/bin/latex2sre",
            "./latex2sre"
        ]
        
        latex2sre_found = False
        for path in latex2sre_paths:
            if path and Path(path).exists():
                self.config.latex2sre_path = path
                latex2sre_found = True
                logger.debug("Found latex2sre at: %s", path)
                break
        
        if not latex2sre_found:
            logger.warning("latex2sre binary not found. Speech generation will be skipped.")

    def validate_latex_batch(self, expressions: List[str]) -> List[Dict[str, Any]]:
        """Validate LaTeX batch using latex-validator."""
        if shutil.which("latex-validator") is None:
            # Return all as valid if tool missing, but mark skipped
            return [{"expression": e, "valid": True, "skipped": True} for e in expressions]
        
        results = []
        for expr in expressions:
            try:
                proc = subprocess.run(
                    ["latex-validator"],
                    input=expr.encode("utf-8"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if proc.returncode == 0:
                    results.append({"expression": expr, "valid": True})
                else:
                    results.append({"expression": expr, "valid": False, "error": proc.stderr.decode("utf-8", errors="ignore")})
            except Exception as e:  # noqa: BLE001
                logger.exception("Exception during latex validation for '%s': %s", expr, e)
                results.append({"expression": expr, "valid": True, "skipped": True})
        
        return results

    def _clean_latex_for_speech(self, expr: str) -> str:
        """Clean LaTeX expression by removing math delimiters for speech generation."""
        # Remove common math delimiters
        expr = expr.strip()
        
        # Remove inline math delimiters: $ ... $
        if expr.startswith('$') and expr.endswith('$'):
            expr = expr[1:-1].strip()
        
        # Remove display math delimiters: $$ ... $$
        if expr.startswith('$$') and expr.endswith('$$'):
            expr = expr[2:-2].strip()
        
        # Remove LaTeX math environment delimiters: \( ... \) and \[ ... \]
        if expr.startswith('\\(') and expr.endswith('\\)'):
            expr = expr[2:-2].strip()
        if expr.startswith('\\[') and expr.endswith('\\]'):
            expr = expr[2:-2].strip()
            
        return expr

    def convert_to_speech_batch(self, expressions: List[str]) -> List[Optional[str]]:
        """Convert LaTeX to spoken text using latex2sre."""
        if not self.config.latex2sre_path or not Path(self.config.latex2sre_path).exists():
            # Tool missing: return None
            return [None for _ in expressions]
        
        latex2sre = self.config.latex2sre_path
        outputs: List[Optional[str]] = []
        for expr in expressions:
            try:
                # Clean the expression by removing dollar signs and other delimiters
                clean_expr = self._clean_latex_for_speech(expr)
                
                proc = subprocess.run(
                    [latex2sre, "--domain", self.config.sre_domain.value, "--locale", self.config.sre_locale, clean_expr],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if proc.returncode == 0:
                    outputs.append(proc.stdout.decode("utf-8").strip())
                else:
                    logger.debug("latex2sre failed for expr: %s, err: %s", clean_expr, proc.stderr.decode("utf-8", errors="ignore"))
                    outputs.append(None)
            except Exception as e:  # noqa: BLE001
                logger.debug("latex2sre exception for expr: %s, err: %s", expr, e)
                outputs.append(None)
        return outputs

    def process_dataset(self) -> ProcessingResult:
        """Main pipeline: clean, validate, convert, save."""
        self._verify_tools()
        stats = ProcessingStats()
        cleaning_agg = CleaningStats()
        output_records: List[Dict[str, Any]] = []

        # Load dataset from Hugging Face datasets
        from datasets import load_dataset  # lazy import

        ds = load_dataset("Kyudan/MathBridge", split="train")
        start = self.config.resume_from or 0
        end = len(ds) if self.config.max_records is None else min(len(ds), start + self.config.max_records)

        batch_size = self.config.batch_size
        pbar = tqdm(range(start, end, batch_size), disable=not self.verbose)

        for i in pbar:
            batch = ds[i : min(i + batch_size, end)]
            # Convert batch dict {key: [values]} to list of dicts [{key: value}]
            records = [
                {key: batch[key][idx] for key in batch.keys()}
                for idx in range(len(next(iter(batch.values()))))
            ]
            batch_stats, cleaning_stats = self._process_batch(records)
            stats.total_processed += batch_stats.get("processed", 0)
            stats.valid_latex += batch_stats.get("valid", 0)
            stats.invalid_latex += batch_stats.get("invalid", 0)
            stats.speech_generated += batch_stats.get("speech_ok", 0)
            stats.speech_failed += batch_stats.get("speech_fail", 0)

            cleaning_agg.merge(cleaning_stats)
            output_records.extend(records)

            if self.verbose:
                pbar.set_description(
                    f"Processed: {stats.total_processed} | valid: {stats.valid_latex} | speech: {stats.speech_generated}"
                )

            # periodic checkpoint
            if len(output_records) >= 10_000:
                self._save_checkpoint(output_records, stats, cleaning_agg)
                output_records = []

        # Save remaining
        files = self._save_final_dataset(output_records, cleaning_agg)
        result = ProcessingResult(config=self.config, stats=stats, output_files=files, errors=[], success=True)
        return result

    def _process_batch(self, records: List[Dict]) -> Tuple[Dict[str, int], 'CleaningStats']:
        """
        Process one batch of records.

        Must:
        - Retain all original dataset keys
        - Add one new key: "sre_spoken_text"
        """
        # Clean
        cleaned_records, clean_counts = self.cleaner.clean_batch(records)
        cleaning_stats = CleaningStats.from_counts(clean_counts)

        # Validate
        exprs = [r.get("equation", "") for r in cleaned_records]
        val_results = self.validate_latex_batch(exprs)
        valid_flags = {res.get("expression", expr): bool(res.get("valid", True)) for res, expr in zip(val_results, exprs)}

        # Convert
        speeches = self.convert_to_speech_batch(exprs)

        stats = {"processed": len(cleaned_records), "valid": 0, "invalid": 0, "speech_ok": 0, "speech_fail": 0}

        for rec, expr, spoken in zip(cleaned_records, exprs, speeches):
            is_valid = valid_flags.get(expr, True)
            if is_valid:
                stats["valid"] += 1
            else:
                stats["invalid"] += 1
            if spoken is not None and spoken != "":
                stats["speech_ok"] += 1
            else:
                stats["speech_fail"] += 1
            # Retain all keys and add sre_spoken_text
            rec["sre_spoken_text"] = spoken

        # mutate original list passed in
        records[:] = cleaned_records
        return stats, cleaning_stats

    def _ensure_output_dir(self) -> Path:
        out = Path(self.config.output_path)
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _save_checkpoint(self, records: List[Dict], stats: ProcessingStats, cleaning_stats: 'CleaningStats'):
        """Save checkpoint file."""
        out_dir = self._ensure_output_dir()
        ckpt_path = out_dir / "checkpoint.jsonl"
        with ckpt_path.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # save stats snapshot
        (out_dir / "checkpoint_stats.json").write_text(json.dumps(stats.dict(), indent=2))
        (out_dir / "checkpoint_cleaning.json").write_text(json.dumps(cleaning_stats.to_dict(), indent=2))
        logger.info("Wrote checkpoint with %d records to %s", len(records), ckpt_path)

    def _save_final_dataset(self, records: List[Dict], cleaning_stats: 'CleaningStats') -> List[str]:
        """Save dataset and cleaning report."""
        out_dir = self._ensure_output_dir()
        files: List[str] = []
        # JSONL
        jsonl_path = out_dir / "mathbridge_processed.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files.append(str(jsonl_path))

        # Parquet
        df = pd.DataFrame.from_records(records)
        parquet_path = out_dir / "mathbridge_processed.parquet"
        df.to_parquet(parquet_path, index=False)
        files.append(str(parquet_path))

        # Cleaning report
        report_path = out_dir / "cleaning_report.json"
        report_path.write_text(json.dumps(cleaning_stats.to_dict(), indent=2))
        files.append(str(report_path))

        logger.info("Saved final dataset with %d records to %s", len(records), out_dir)
        return files
