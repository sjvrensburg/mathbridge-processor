# mathbridge-processor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

A command-line tool for processing the MathBridge dataset with LaTeX validation, speech generation, and data cleaning capabilities.

## Features

- 🔍 **LaTeX Validation**: Validates mathematical expressions using latex-validator
- 🗣️ **Speech Generation**: Converts LaTeX to speech text using Speech Rule Engine (SRE)  
- 🧹 **Data Cleaning**: Cleans and normalizes dataset entries
- 📊 **Multiple Output Formats**: Supports JSONL and Parquet output formats
- 🔄 **Batch Processing**: Efficient batch processing with progress tracking
- ⚡ **Parallel Processing**: Multi-threaded validation and speech conversion for optimal performance
- 🎯 **Intelligent Caching**: Expression-level caching reduces redundant processing
- 🔧 **Recovery Tool**: Rebuild complete output files from checkpoint data without reprocessing
- ⚙️ **Configurable**: Flexible configuration via JSON config files or CLI arguments

## Installation

### Using pipx (Recommended)

Install mathbridge-processor in an isolated environment using [pipx](https://pypa.github.io/pipx/):

```bash
pipx install git+https://github.com/sjvrensburg/mathbridge-processor.git
```

### Using pip

```bash
pip install git+https://github.com/sjvrensburg/mathbridge-processor.git
```

### Development Installation

```bash
git clone https://github.com/sjvrensburg/mathbridge-processor.git
cd mathbridge-processor
pip install -e .
```

## Usage

### Quick Start

Process 100 records from the MathBridge dataset:

```bash
mathbridge-process process --max-records 100 --verbose
```

### Configuration

Create a configuration file:

```bash
mathbridge-process create-config
```

This creates a `config.json` file that you can customize:

```json
{
  "sre_domain": "clearspeak",
  "sre_locale": "en", 
  "batch_size": 1000,
  "max_records": null,
  "resume_from": 0,
  "output_path": "mathbridge_processed",
  "latex2sre_path": "./latex2sre"
}
```

Run with custom configuration:

```bash
mathbridge-process process --config config.json
```

### Command Line Options

```bash
mathbridge-process process --help
```

Available options:
- `--output`: Output directory (default: mathbridge_processed)
- `--config`: Path to JSON config file
- `--batch-size`: Batch size for processing
- `--max-records`: Maximum number of records to process
- `--resume-from`: Resume processing from specific index
- `--sre-domain`: SRE domain (mathspeak|clearspeak)
- `--sre-locale`: SRE locale (default: en)
- `--latex2sre-path`: Path to latex2sre binary
- `--max-workers`: Maximum parallel workers (default: auto-detect)
- `--verbose`: Enable verbose output

### Recovery Tool

If your processing completed but you only have partial output files (due to the checkpointing issue fixed in v0.1.1), you can recover the complete dataset from checkpoint data:

```bash
mathbridge-process recover <checkpoint_directory> --verbose
```

This command:
- Reads all processed data from the checkpoint file
- Rebuilds complete JSONL and Parquet output files  
- Completes in minutes instead of hours/days of reprocessing
- Preserves all original processing statistics

**Example:**
```bash
# Recover from existing checkpoint data
mathbridge-process recover mathbridge_processed --verbose
```

### Other Commands

Show AI agent usage instructions:
```bash
mathbridge-process agent-info
```

## Dependencies

The tool requires:
- Python 3.9+
- latex-validator (for LaTeX validation)
- latex2sre (optional, for speech generation)

Speech generation requires the `latex2sre` binary. If not found, the tool will skip speech generation but continue processing.

## Output

The tool generates:
- `mathbridge_processed.jsonl`: Processed dataset in JSONL format
- `mathbridge_processed.parquet`: Processed dataset in Parquet format  
- `cleaning_report.json`: Report of cleaning operations performed
- `checkpoint.jsonl`: Incremental processing checkpoint (for recovery)

Each output record contains all original dataset columns plus:
- `sre_spoken_text`: Generated speech text (if available)

## Important Fixes (v0.1.1)

### Critical Data Loss Bug Fixed

**Issue**: Previous versions had a critical bug where only the final batch of records (~5,000-10,000) was saved to output files, despite processing all records successfully. This caused 99%+ data loss in the final output.

**Root Cause**: The checkpointing logic incorrectly cleared the output buffer every 10,000 records, keeping only the last partial batch.

**Fix**: 
- Updated final dataset generation to read all checkpoint data
- Added recovery tool to rebuild output files from existing checkpoints
- All processed records are now correctly included in final output

**Impact**: Users who processed large datasets with v0.1.0 can use the `recover` command to get their complete results without reprocessing.

## Performance Optimization

### Recommended Settings for Large Datasets

For optimal performance when processing large datasets (tested on 20-core i7-12700H system):

```bash
mathbridge-process process \
  --batch-size 1000 \
  --max-workers 20 \
  --verbose
```

### Performance Benchmarks

| Dataset Size | Processing Time | Records/Second | Configuration |
|--------------|----------------|----------------|---------------|
| 500 records  | 10.8s         | 46.1 rec/s     | batch=500, workers=20 |
| 2,000 records | 27.6s        | 72.6 rec/s     | batch=1000, workers=20 |
| 5,000 records | 58.7s        | 85.2 rec/s     | batch=1000, workers=20 |
| 8,000 records | 88.4s        | 90.5 rec/s     | batch=1000, workers=20 |

### Large-Scale Processing Estimates

For processing 23 million records:
- **Estimated time**: 74-88 hours (3-4 days)
- **Peak performance**: ~90 records/second
- **Storage requirements**: ~21GB output, 43GB recommended free space
- **Caching benefits**: Performance improves with larger datasets due to expression reuse

### Optimization Tips

1. **Worker Count**: Use `--max-workers` equal to your CPU core count for best throughput
2. **Batch Size**: Use `--batch-size 1000` for optimal memory/performance balance  
3. **Storage**: Use SSD storage for output directory when processing large datasets
4. **Memory**: Ensure sufficient RAM (8GB+ recommended for large datasets)
5. **Chunked Processing**: For maximum safety, process in 1-5M record chunks using `--max-records` and `--resume-from`

### Environment Variables

Set these for consistent performance tuning:
- `MB_MAX_WORKERS`: Default worker count
- `MB_BATCH_SIZE`: Default batch size
- `MB_LATEX2SRE_PATH`: Path to latex2sre binary

Example:
```bash
export MB_MAX_WORKERS=20
export MB_BATCH_SIZE=1000
export MB_LATEX2SRE_PATH=/usr/local/bin/latex2sre
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.