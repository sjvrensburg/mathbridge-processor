import json
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self._speech_cache: Dict[str, Optional[str]] = {}  # Cache for speech generation
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
        
        # Use parallel processing for validation
        if self.config.max_workers != 1 and len(expressions) > 1:
            return self._validate_latex_parallel(expressions)
        
        # Fallback to sequential processing
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
    
    def _validate_latex_parallel(self, expressions: List[str]) -> List[Dict[str, Any]]:
        """Validate LaTeX expressions in parallel using ThreadPoolExecutor."""
        def validate_single(expr: str) -> Dict[str, Any]:
            try:
                proc = subprocess.run(
                    ["latex-validator"],
                    input=expr.encode("utf-8"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if proc.returncode == 0:
                    return {"expression": expr, "valid": True}
                else:
                    return {"expression": expr, "valid": False, "error": proc.stderr.decode("utf-8", errors="ignore")}
            except Exception as e:  # noqa: BLE001
                logger.debug("Exception during latex validation for '%s': %s", expr, e)
                return {"expression": expr, "valid": True, "skipped": True}
        
        # Determine optimal number of workers
        max_workers = self.config.max_workers or min(32, len(expressions), (os.cpu_count() or 1) + 4)
        
        results = [None] * len(expressions)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all validation tasks
            future_to_index = {
                executor.submit(validate_single, expr): i 
                for i, expr in enumerate(expressions)
            }
            
            # Collect results maintaining original order
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.debug("Parallel validation exception for index %d: %s", index, e)
                    results[index] = {"expression": expressions[index], "valid": True, "skipped": True}
        
        logger.debug(f"Parallel validation completed with {max_workers} workers for {len(expressions)} expressions")
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
        """Convert LaTeX to spoken text using latex2sre with caching and batch processing."""
        if not self.config.latex2sre_path or not Path(self.config.latex2sre_path).exists():
            # Tool missing: return None
            return [None for _ in expressions]
        
        # Clean all expressions first
        cleaned_expressions = [self._clean_latex_for_speech(expr) for expr in expressions]
        
        # Check cache first and separate cached vs uncached expressions
        results: List[Optional[str]] = [None] * len(expressions)
        uncached_indices: List[int] = []
        uncached_expressions: List[str] = []
        
        for i, clean_expr in enumerate(cleaned_expressions):
            if clean_expr in self._speech_cache:
                results[i] = self._speech_cache[clean_expr]
                logger.debug(f"Cache hit for expression: {clean_expr[:50]}...")
            else:
                uncached_indices.append(i)
                uncached_expressions.append(clean_expr)
        
        cache_hits = len(expressions) - len(uncached_expressions)
        if cache_hits > 0:
            logger.debug(f"Speech cache hits: {cache_hits}/{len(expressions)}")
        
        # Process uncached expressions if any
        if uncached_expressions:
            # Choose processing strategy based on batch size and configuration
            if self.config.max_workers != 1 and len(uncached_expressions) <= 20:
                # Use parallel processing for small batches
                batch_results = self._convert_to_speech_parallel(uncached_expressions)
            else:
                # Use batch file processing for larger batches
                batch_results = self._convert_to_speech_batch_file(uncached_expressions)
            
            # Update cache and results
            for idx, batch_idx in enumerate(uncached_indices):
                speech_result = batch_results[idx] if idx < len(batch_results) else None
                clean_expr = uncached_expressions[idx]
                
                # Cache the result (including None for failed conversions)
                self._speech_cache[clean_expr] = speech_result
                results[batch_idx] = speech_result
        
        return results
    
    def _convert_to_speech_batch_file(self, expressions: List[str]) -> List[Optional[str]]:
        """Convert expressions using latex2sre batch file processing."""
        if not expressions:
            return []
            
        latex2sre = self.config.latex2sre_path
        
        try:
            # Create temporary input file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as input_file:
                for expr in expressions:
                    input_file.write(f"{expr}\n")
                input_file_path = input_file.name
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as output_file:
                output_file_path = output_file.name
            
            # Run latex2sre in batch mode
            proc = subprocess.run(
                [
                    latex2sre,
                    "--domain", self.config.sre_domain.value,
                    "--locale", self.config.sre_locale,
                    "--input", input_file_path,
                    "--output", output_file_path,
                    "--stream"  # Stream output line by line
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            
            # Read results
            results: List[Optional[str]] = []
            if proc.returncode == 0:
                try:
                    with open(output_file_path, 'r') as f:
                        for line in f:
                            speech_text = line.strip()
                            results.append(speech_text if speech_text else None)
                    
                    # Ensure we have the same number of results as expressions
                    while len(results) < len(expressions):
                        results.append(None)
                        
                    logger.debug(f"Batch processed {len(expressions)} expressions, got {len(results)} results")
                    
                except Exception as e:
                    logger.error(f"Error reading batch output file: {e}")
                    results = [None] * len(expressions)
            else:
                logger.debug(f"latex2sre batch processing failed: {proc.stderr.decode('utf-8', errors='ignore')}")
                results = [None] * len(expressions)
            
            # Clean up temporary files
            try:
                os.unlink(input_file_path)
                os.unlink(output_file_path)
            except OSError as e:
                logger.debug(f"Could not delete temporary files: {e}")
            
            return results
            
        except Exception as e:
            logger.exception(f"Exception during batch speech conversion: {e}")
            return [None] * len(expressions)

    def _convert_to_speech_parallel(self, expressions: List[str]) -> List[Optional[str]]:
        """Convert expressions using parallel latex2sre individual calls."""
        if not expressions:
            return []
        
        latex2sre = self.config.latex2sre_path
        
        def convert_single(expr: str) -> Optional[str]:
            try:
                proc = subprocess.run(
                    [latex2sre, "--domain", self.config.sre_domain.value, "--locale", self.config.sre_locale, expr],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if proc.returncode == 0:
                    result = proc.stdout.decode("utf-8").strip()
                    return result if result else None
                else:
                    logger.debug(f"latex2sre failed for expr: {expr[:50]}..., err: {proc.stderr.decode('utf-8', errors='ignore')}")
                    return None
            except Exception as e:
                logger.debug(f"latex2sre exception for expr: {expr[:50]}..., err: {e}")
                return None
        
        # Determine optimal number of workers
        max_workers = self.config.max_workers or min(8, len(expressions), (os.cpu_count() or 1))
        
        results = [None] * len(expressions)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all conversion tasks
            future_to_index = {
                executor.submit(convert_single, expr): i 
                for i, expr in enumerate(expressions)
            }
            
            # Collect results maintaining original order
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.debug("Parallel speech conversion exception for index %d: %s", index, e)
                    results[index] = None
        
        logger.debug(f"Parallel speech conversion completed with {max_workers} workers for {len(expressions)} expressions")
        return results

    def get_cache_stats(self) -> Dict[str, int]:
        """Get speech generation cache statistics."""
        total_cached = len(self._speech_cache)
        successful_cached = sum(1 for result in self._speech_cache.values() if result is not None)
        failed_cached = total_cached - successful_cached
        
        return {
            "total_expressions_cached": total_cached,
            "successful_cached": successful_cached,
            "failed_cached": failed_cached
        }
        
    def clear_cache(self):
        """Clear the speech generation cache."""
        cache_size = len(self._speech_cache)
        self._speech_cache.clear()
        logger.debug(f"Cleared speech cache containing {cache_size} expressions")

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

        # Save remaining records + checkpoint file as final dataset
        files = self._save_final_dataset_from_checkpoint(output_records, cleaning_agg)
        
        # Include cache statistics
        cache_stats = self.get_cache_stats()
        
        result = ProcessingResult(
            config=self.config, 
            stats=stats, 
            output_files=files, 
            errors=[], 
            success=True,
            cache_stats=cache_stats
        )
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

    def _save_final_dataset_from_checkpoint(self, remaining_records: List[Dict], cleaning_stats: 'CleaningStats') -> List[str]:
        """Save final dataset by reading all records from checkpoint file + remaining records."""
        out_dir = self._ensure_output_dir()
        files: List[str] = []
        
        # Read all records from checkpoint file + add any remaining records
        all_records = []
        
        # First, read all records from checkpoint if it exists
        ckpt_path = out_dir / "checkpoint.jsonl"
        if ckpt_path.exists():
            with ckpt_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_records.append(json.loads(line.strip()))
        
        # Add any remaining records not yet checkpointed
        all_records.extend(remaining_records)
        
        # JSONL
        jsonl_path = out_dir / "mathbridge_processed.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files.append(str(jsonl_path))

        # Parquet (only if we have records to avoid empty DataFrame issues)
        if all_records:
            df = pd.DataFrame.from_records(all_records)
            parquet_path = out_dir / "mathbridge_processed.parquet"
            df.to_parquet(parquet_path, index=False)
            files.append(str(parquet_path))

        # Cleaning report
        report_path = out_dir / "cleaning_report.json"
        report_path.write_text(json.dumps(cleaning_stats.to_dict(), indent=2))
        files.append(str(report_path))

        logger.info("Saved final dataset with %d total records (%d from checkpoint + %d remaining) to %s", 
                   len(all_records), len(all_records) - len(remaining_records), len(remaining_records), out_dir)
        return files
    
    def _save_final_dataset(self, records: List[Dict], cleaning_stats: 'CleaningStats') -> List[str]:
        """Save dataset and cleaning report (legacy method - kept for backward compatibility)."""
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
