# src/data

**Note: This is an LLM-generated doc with light human review.**

This module handles data formatting for Supervised Fine-Tuning (SFT). It takes raw workflow JSON files from ToolBench and converts them into OpenAI-style SFT samples, with support for negative (irrelevance) sample generation.

## Pipeline Overview

```
raw workflow JSONs  →  [format]  →  positive SFT samples
                    →  [negatives]  →  negative SFT samples
                    →  [split]  →  train.jsonl / test.jsonl
```

## Modules

| File                     | Description                                                                                                |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `data_format_main.py`    | **Main entrypoint.** Runs the full pipeline: format → negatives → split.                                   |
| `data_format_lib.py`     | Core formatting logic. Converts raw workflows into SFT samples.                                            |
| `negative_sample_lib.py` | Generates negative (irrelevance) samples where no relevant tools are in the candidate pool.                |
| `split_lib.py`           | Splits formatted samples into train/test sets and combines them into JSONL.                                |
| `prompt_template.py`     | System prompt and response schema templates.                                                               |
| `toolbench/`             | Scripts for preprocessing raw ToolBench data (format conversion, token filtering, chat template analysis). |

## Running the Full Pipeline

```bash
python src/data/data_format_main.py \
    --workspace_dir $workspace_dir \
    --num_tools 10 \
    --test_ratio 0.0 \
    --negative_ratio 0.2 \
    --output_format python
```

**Key arguments:**

- `--workspace_dir`: Root dir. Raw data expected at `$workspace_dir/data/`. Output goes to `$workspace_dir/`.
- `--raw_data_dir`: Override raw data directory (optional).
- `--num_tools`: Number of tool candidates per sample (positives + negatives).
- `--negative_ratio`: Fraction of raw samples to use for negative generation.
- `--in_place_neg`: If set, negatives are sampled from the raw data and excluded from positives (strict split).
- `--num_processes`: Number of parallel workers (default: 16).

## Toolbench Preprocessing (optional but recommended)

Convert raw ToolBench data to TRL format:

```bash
python src/data/toolbench/format_toolbench_to_trl.py \
    --input_path /path/to/input.json \
    --output_path /path/to/output.jsonl \
    --tokenizer_name google/gemma-3-4b-it \
    --chat_template src/data/toolbench/gemma_custom.jinja
```

Filter samples that exceed a token length:

```bash
python src/data/toolbench/filter_long_samples.py \
    --input_path /path/to/input.jsonl \
    --output_path /path/to/output.jsonl \
    --max_length 8000 \
    --tokenizer_name google/gemma-3-4b-it
```
