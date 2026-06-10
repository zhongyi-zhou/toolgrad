# ToolGrad: Efficient Tool-use Dataset Generation with Textual "Gradients" (ACL 2026 Finding)
<!--- BADGES: START --->
[![GitHub license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)][#license]
[![Arxiv](https://img.shields.io/badge/arXiv-2508.04086-B31B1B.svg)][#arxiv-paper] 
[![PyPI](https://img.shields.io/static/v1?label=PyPI&message=toolgrad&color=lightgrey)][#pypi-package] 
[![Dataset on HF](https://img.shields.io/badge/Dataset-on%20Hugging%20Face-FF6C37?logo=HuggingFace)][#dataset-hf]
[![ToolGrad-1B on HF](https://img.shields.io/badge/ToolGrad--1B-on%20Hugging%20Face-FF6C37?logo=HuggingFace)][#model-1b-hf]
[![ToolGrad-4B on HF](https://img.shields.io/badge/ToolGrad--4B-on%20Hugging%20Face-FF6C37?logo=HuggingFace)][#model-4b-hf]
[![ToolGrad-12B on HF](https://img.shields.io/badge/ToolGrad--12B-on%20Hugging%20Face-FF6C37?logo=HuggingFace)][#model-12b-hf]

[#pypi-package]: https://pypi.org/project/toolgrad
[#license]: LICENSE
[#arxiv-paper]: http://arxiv.org/abs/2508.04086
[#dataset-hf]: https://huggingface.co/datasets/zhongyi-zhou/toolgrad-500
[#model-1b-hf]: https://huggingface.co/zhongyi-zhou/toolgrad-1b
[#model-4b-hf]: https://huggingface.co/zhongyi-zhou/toolgrad-4b
[#model-12b-hf]: https://huggingface.co/zhongyi-zhou/toolgrad-12b
<!--- BADGES: END --->

This is the official repository for **ToolGrad: Efficient Tool-use Dataset Generation with Textual "Gradients"**.

<p align="center">
  <img src="images/teaser-1.jpg" alt="demo"/>
</p>

<p align="center">
  <img src="images/table.png" alt="demo"/>
</p>

---

## Part I: A Quick ToolGrad Framework Demo


> **No GPU required.** **No ToolBench API Key required.**

### 1. Setup Environment
Clone the repository and initialize the Python virtual environment:
```bash
git clone https://github.com/zhongyi-zhou/toolgrad.git
cd toolgrad
uv venv
uv sync
source .venv/bin/activate
```

### 2. Launch ToolGrad MCP Quick Start
To run ToolGrad on a Model Context Protocol (MCP) filesystem service:

1. Install Node.js/npx (required for MCP service):
   ```bash
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
   nvm install 20
   nvm alias default 20
   ```
2. Launch the demo execution (requires a Gemini API key):
   ```bash
   export GEMINI_API_KEY=YOUR_GEMINI_KEY
   export PYTHONPATH=./
   python examples/mcp_filesystem.py
   ```
   During the execution, you should be able to see some useful logs to better understand the ToolGrad framework.

    For example, the following is an example log of the API proposer:
    ```bash
    INFO:root:[Iteration 2] Proposed 3 API proposals
    INFO:root:  Proposal 1 (proposal_1): Read the content of the 'favorite_books.txt' file.
    INFO:root:    └─ read_text_file
    INFO:root:  Proposal 2 (proposal_2): Read the contents of 'favorite_books.txt', 'favorite_cities.txt', and 'favorite_songs.txt' simultaneously.
    INFO:root:    └─ read_multiple_files
    INFO:root:  Proposal 3 (proposal_3): List the contents of the current directory.
    INFO:root:    └─ list_directory
    ```

    The following is an example log of the API executor:
    ```bash
    INFO:root:[Iteration 2] Executed 3 proposals: 3 successful, 0 failed
    INFO:root:  ✓ proposal_1: 1 tool call(s)
    INFO:root:      Tool: read_text_file
    INFO:root:      Input: {'path': 'favorite_books.txt'}
    INFO:root:  ✓ proposal_2: 1 tool call(s)
    INFO:root:      Tool: read_multiple_files
    INFO:root:      Input: {'paths': ['favorite_books.txt', 'favorite_cities.txt', 'favorite_songs.txt']}
    INFO:root:  ✓ proposal_3: 1 tool call(s)
    INFO:root:      Tool: list_directory
    INFO:root:      Input: {'path': '.'}
    ```

    After execution, you should be able to see the output data in the [examples/outputs/](examples/outputs/) folder (which will look similar to [examples/outputs/trace_example/00123.json](examples/outputs/trace_example/00123.json) and [examples/outputs/example_seed=123__iter=3__num_apis=5.json](examples/outputs/example_seed=123__iter=3__num_apis=5.json)).

    *If the demo works for you, please consider giving us a star! ⭐*

---

## Part II: Reproduction

> This part **requires a GPU** (verified on a single NVIDIA A100-40GB GPU).

This section details how to reproduce our evaluation scores on the BFCL V1 & V2 benchmark and how to post-train Gemma 3 models on the ToolGrad-500 dataset.

### 1. Reproducing BFCL Evaluation Results
To evaluate ToolGrad models on the Berkeley Function Calling Leaderboard (BFCL) V1 & V2:

#### Step 1.1: Setup BFCL Submodule
The evaluation framework uses our customized fork of BFCL. Initialize the submodule:
```bash
git submodule update --init --recursive
```

#### Step 1.2: Run Evaluation inside Docker
Local inference is run inside a GPU-enabled Docker container using the official `vllm/vllm-openai:v0.22.1-cu129` image. 

> [!NOTE]
> This evaluation setup has been verified on a single **NVIDIA A100-SXM4-40GB** GPU.

Run the following command from the root of the repository to launch the container and execute the evaluation script (optionally passing the target model `toolgrad_1b`, `toolgrad_4b`, or `toolgrad_12b` as the first argument, which defaults to `toolgrad_1b`):

```bash
docker run --rm --entrypoint bash \
  --device /dev/nvidia0 --device /dev/nvidia-uvm --device /dev/nvidiactl \
  -v /usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  -v /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
  -v $(pwd):$(pwd) \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.22.1-cu129 $(pwd)/src/scripts/run_bfcl_eval.sh toolgrad_1b
```

> [!NOTE]
> If model weights are not found locally in the mounted Hugging Face cache folder, they will be automatically downloaded from Hugging Face Hub.
>
> To evaluate a custom local directory checkpoint instead of downloading from HF, pass the `--local-model-path` flag as an additional argument:
> ```bash
> ... src/scripts/run_bfcl_eval.sh toolgrad_1b --local-model-path /path/to/local/model/dir
> ```

#### Step 1.3: View Evaluation Scores
Once completed, view the generated responses and scores under the submodule directory (`ext/gorilla-for-toolgrad/berkeley-function-call-leaderboard/`):
- **Generated Responses**: saved in the `result/` folder.
- **Accuracy Scores & CSV Summaries**: saved in the `score/` folder (e.g., `score/data_overall.csv`).

---

### 2. Supervised Fine-Tuning (SFT) Gemma 3

Post-train Gemma 3 models on the [ToolGrad-500 dataset](https://huggingface.co/datasets/zhongyi-zhou/toolgrad-500).

#### Step 2.1: Setup Training
Ensure you have activated your virtual environment:
```bash
source .venv/bin/activate
```

#### Step 2.2: Launch SFT on ToolGrad-1B
To run Supervised Fine-Tuning on `google/gemma-3-1b-it` (reproducing global batch size 2 via gradient accumulation on a single GPU):

```bash
python src/train/train_sft.py \
  --model google/gemma-3-1b-it \
  --learning_rate 1e-5 \
  --num_epochs 3 \
  --seq_length 8192 \
  --gradient_checkpointing \
  --gradient_accumulation_steps 2
```

> [!NOTE]
> For SFT training recipes of other model configurations (such as the 4B and 12B models) or training with custom local datasets, please refer to the detailed [src/train/README.md](src/train/README.md) guide.

---

## Part III: Generating your own ToolGrad-500


> This part **requires a ToolBench API key**, which may take time to be issued.

### 1. Generating Raw ToolGrad Workflows
To generate the raw workflow files, we execute the `src/generate_toolgrad_data.py` script.

First, obtain and export your `TOOLBENCH_KEY` (you can apply for an API key by following the instructions on the [ToolBench repository](https://github.com/openbmb/toolbench)):
```bash
export TOOLBENCH_KEY=YOUR_TOOLBENCH_API_KEY
```

You also need to set up the ToolBench API database:
1. Download `tools.zip` from [Google Drive](https://drive.google.com/file/d/1pM161RiqwEdE6L-kaTS4P0OpYB2I_Phl/view?usp=sharing) and extract it to reveal a `tools/` folder.
2. Set the `TOOLBENCH_LIBRARY_ROOT` environment variable pointing to the extracted path:
```bash
export TOOLBENCH_LIBRARY_ROOT=/path/to/extracted/tools
```

To run a single workflow generation task for a specific seed:
```bash
python src/generate_toolgrad_data.py \
    --cfg examples/configs/gemini-2.5-lite.gin \
    --iter 10 \
    --num_apis 50 \
    --output_dir /path/to/output_dir \
    --seed 123
```

To generate the full 500 workflow samples in parallel, we provide the `src/scripts/generate_toolgrad_500.sh` helper script (make sure to adjust the configuration, parallel worker count, and target output directory inside the script as needed):
```bash
bash src/scripts/generate_toolgrad_500.sh
```

### 2. Pipeline Overview
```
raw workflow JSONs  →  [format]  →  positive SFT samples
                    →  [negatives]  →  negative SFT samples
                    →  [split]  →  train.jsonl / test.jsonl
```

### 3. Format & Split Raw Data
Execute the formatting and negative sample generation pipeline on the raw workflows:
```bash
python src/data/data_format_main.py \
    --workspace_dir $(pwd) \
    --num_tools 10 \
    --test_ratio 0.0 \
    --negative_ratio 0.2 \
    --output_format python
```

*   `--workspace_dir`: Root directory. Raw workflow data is expected at `$(pwd)/data/`.
*   `--num_tools`: Number of tool candidates per SFT sample.
*   `--negative_ratio`: Fraction of raw samples to use for generating negative tool use cases.

### 4. Toolbench Preprocessing (Optional)
To convert raw ToolBench dataset formats to TRL formats:
```bash
python src/data/toolbench/format_toolbench_to_trl.py \
    --input_path /path/to/input.json \
    --output_path /path/to/output.jsonl \
    --tokenizer_name google/gemma-3-4b-it \
    --chat_template src/data/toolbench/gemma_custom.jinja
```

Filter SFT samples exceeding 8k context limits:
```bash
python src/data/toolbench/filter_long_samples.py \
    --input_path /path/to/input.jsonl \
    --output_path /path/to/output.jsonl \
    --max_length 8000 \
    --tokenizer_name google/gemma-3-4b-it
```

---

## BibTeX Citation
```bibtex
@misc{zhou2025toolgradefficienttoolusedataset,
      title={ToolGrad: Efficient Tool-use Dataset Generation with Textual "Gradients"}, 
      author={Zhongyi Zhou and Kohei Uehara and Haoyu Zhang and Jingtao Zhou and Lin Gu and Ruofei Du and Zheng Xu and Tatsuya Harada},
      year={2025},
      eprint={2508.04086},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.04086}, 
}
```