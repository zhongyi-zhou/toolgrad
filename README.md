# ToolGrad: Efficient Tool-use Dataset Generation with Textual "Gradients" (ACL 26 Finding)
<!--- BADGES: START --->
[![GitHub license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)][#license]
[![Arxiv](https://img.shields.io/badge/arXiv-2508.04086-B31B1B.svg)][#arxiv-paper] 
[![PyPI](https://img.shields.io/static/v1?label=PyPI&message=toolgrad&color=lightgrey)][#pypi-package] 
[![Dataset on HF](https://img.shields.io/badge/Dataset-on%20Hugging%20Face-FF6C37?logo=HuggingFace)][#dataset-hf]
[![ToolGrad-1B on HF](https://img.shields.io/badge/ToolGrad--1B-on%20Hugging%20Face-FF6C37?logo=HuggingFace)][#model-1b-hf]
[![ToolGrad-4B on HF](https://img.shields.io/badge/ToolGrad--4B-on%20Hugging%20Face-FF6C37?logo=HuggingFace)][#model-4b-hf]
[![ToolGrad-12B on HF](https://img.shields.io/badge/ToolGrad--12B-on%20Hugging%20Face-FF6C37?logo=HuggingFace)][#model-12b-hf]
<!-- Replace the PyPI badge with ToolGrad later -->

> [!WARNING]
> **Notes**: The current repo contains out-of-dated code. We will actively update the repo to align with our paper on ACL 2026. The following is a roadmap.
> 
> - [x] Update data generation pipeline
> - [x] Update toolgrad package
> - [ ] post-training code
> - [ ] evaluate code

<!-- Replace the PyPI link with ToolGrad later -->
[#pypi-package]: https://pypi.org/project/toolgrad
[#license]: LICENSE
[#arxiv-paper]: http://arxiv.org/abs/2508.04086

[#dataset-hf]: https://huggingface.co/datasets/zhongyi-zhou/toolgrad-500
[#model-1b-hf]: https://huggingface.co/zhongyi-zhou/toolgrad-1b
[#model-4b-hf]: https://huggingface.co/zhongyi-zhou/toolgrad-4b
[#model-12b-hf]: https://huggingface.co/zhongyi-zhou/toolgrad-12b

<!--- BADGES: END --->

This is an official repo for <ToolGrad: Efficient Tool-use Dataset Generation with Textual “Gradients”>.

<p align="center">
  <img src="images/teaser-1.jpg" alt="demo"/>
</p>

<p align="center">
  <img src="images/table.png" alt="demo"/>
</p>

## Get Started: A Quick Demo

### Step 0: Install packages

Install `nvm` for MCP service:
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
nvm install 20
nvm alias default 20

# verify npx
npx --version
```

Clone this repo and setup the uv env:

```bash
cd toolgrad
uv venv
uv sync
source .venv/bin/activate
```

### Step 1: launch your first ToolGrad framework on a MCP service

```bash
# If you have no Gemini API key in your environment,
export GEMINI_API_KEY=YOUR_GEMINI_KEY

export PYTHONPATH=./
python examples/mcp_filesystem.py
```

During the execution, you should be able to see some useful logs to better understand the ToolGrad framework.

For example, the following is an example log of the API proposer.
```bash
INFO:root:[Iteration 2] Proposed 3 API proposals
INFO:root:  Proposal 1 (proposal_1): Read the content of the 'favorite_books.txt' file.
INFO:root:    └─ read_text_file
INFO:root:  Proposal 2 (proposal_2): Read the contents of 'favorite_books.txt', 'favorite_cities.txt', and 'favorite_songs.txt' simultaneously.
INFO:root:    └─ read_multiple_files
INFO:root:  Proposal 3 (proposal_3): List the contents of the current directory.
INFO:root:    └─ list_directory
```

The following is an example log of API executor.
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

After the execution, you should be able to see the output data in [examples/outputs/](examples/outputs/). It should look similar to [examples/outputs/trace_example/00123.json](examples/outputs/trace_example/00123.json) and [examples/outputs/example_seed=123__iter=3__num_apis=5.json](examples/outputs/example_seed=123__iter=3__num_apis=5.json).

## ToolGrad-500 and ToolGrad Models
We release our dataset, model checkpoints, and collection on Hugging Face:
- **Hugging Face Collection:** [zhongyi-zhou/toolgrad](https://huggingface.co/collections/zhongyi-zhou/toolgrad)
- **Dataset:** [zhongyi-zhou/toolgrad-500](https://huggingface.co/datasets/zhongyi-zhou/toolgrad-500)
- **Model Checkpoints:**
  - [ToolGrad-1B](https://huggingface.co/zhongyi-zhou/toolgrad-1b)
  - [ToolGrad-4B](https://huggingface.co/zhongyi-zhou/toolgrad-4b)
  - [ToolGrad-12B](https://huggingface.co/zhongyi-zhou/toolgrad-12b)



## BibTex
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