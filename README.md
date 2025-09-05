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
- `--verbose`: Enable verbose output

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

Each output record contains all original dataset columns plus:
- `sre_spoken_text`: Generated speech text (if available)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.