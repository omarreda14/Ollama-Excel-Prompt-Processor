# Ollama Excel Prompt Processor

A Python script that reads an Excel file of prompts, sends them to one or more models via the Ollama API, and writes the responses back to an Excel file. It supports both standard models and those with a "thinking" mode (removing `<think>...</think>` tags to output only the final answer).

## Features

- **Excel I/O:** Reads prompts from an `.xlsx` file and writes responses into separate columns.
- **Multi-Model Support:** Process prompts with multiple models at once.
- **Thinking Mode Handling:** Automatically removes `<think>...</think>` sections from models that output chain-of-thought responses.
- **Batch Processing:** Saves progress periodically to avoid data loss.
- **Rate Limiting:** Configurable delay between API calls.

## Requirements

- Python 3.6+
- Python packages: `pandas`, `requests`, `openpyxl`

Install the dependencies with:

```bash
pip install pandas requests openpyxl
