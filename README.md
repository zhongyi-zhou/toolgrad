# ToolGrad: Efficient Tool-use Dataset Generation with Textual "Gradients"
<!--- BADGES: START --->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#colab)
[![GitHub license](https://img.shields.io/badge/License-CC--BY%204.0-blue.svg)][#license]
[![Arxiv](https://img.shields.io/badge/arXiv-2508.04086-B31B1B.svg)][#arxiv-paper] 
[![PyPI](https://img.shields.io/static/v1?label=PyPI&message=toolgrad&color=lightgrey)][#pypi-package] 
<!-- Replace the PyPI badge with ToolGrad later -->

TODOs:
[![Dataset on HF](https://img.shields.io/badge/Dataset-on%20Hugging%20Face-FF6C37?logo=HuggingFace)][#dataset-hf]
[![Model on HF](https://img.shields.io/badge/Model-on%20Hugging%20Face-FF6C37?logo=HuggingFace)][#model-hf]


<!-- Replace the PyPI link with ToolGrad later -->
[#pypi-package]: https://pypi.org/project/toolgrad
[#license]: https://img.shields.io/badge/license-CC--BY--NC%204.0-blue
[#arxiv-paper]: http://arxiv.org/abs/2508.04086

[#dataset-hf]: https://huggingface.co/datasets/
[#model-hf]: https://huggingface.co/models/

<!--- BADGES: END --->

This is an official repo for <ToolGrad: Efficient Tool-use Dataset Generation with Textual “Gradients”>.

## Get Started: A Quick Demo

### Step 0: Install packages

```bash
git clone https://github.com/zhongyi-zhou/toolgrad.git
cd toolgrad
conda env create -f environment.yml
conda activate toolgrad
```

### Step 1: launch your first ToolGrad framework on a MCP service

```bash
export PYTHONPATH=./
python examples/mcp_filesystem.py
```

## Reproduction of Dataset Generation
### Step 0: ToolBench API Key
You need to first obtain a ToolBench API key by following their instruction:
- https://github.com/OpenBMB/ToolBench


Note: The API key is necessary for the following procedures.
### Step 1: ToolBench Setups
```bash
export TOOLBENCH_KEY=YOURTOOLBENCHKEY
```

You also need to setup the ToolBench API database:
- Unzip `tools.zip` ([Google Drive](https://drive.google.com/file/d/1pM161RiqwEdE6L-kaTS4P0OpYB2I_Phl/view?usp=sharing)) and it will show a `tools/` folder.
- add this path to the environ as follow
```bash
export TOOLBENCH_LIBRARY_ROOT=YOUR_PATH/TO/TOOLS
```


### Step 2: Generate your first ToolGrad sample on the ToolBench API database

```bash
export PYTHONPATH=./
python examples/toolbench.py
```
You will then find a new json file under `examples/outputs/`. `examples/example_outputs/seed=123__iter=5__num_apis=50.json` is an example that we generated.

ToolGrad-5K is composed of 5k data generation sessions with different seed.
It takes ~250 USD to generate the full 5K dataset, using gpt-4.1-mini.
## Evaluation

First download the dataset from [Google Drive](https://drive.google.com/file/d/1fogq9N9P02I0SIycnDjjdLDC4TmR4BCz/view?usp=sharing) and unzip it.
You should be able see a folder structure as follows:
```
ToolGrad-5k  
├── data  
├── metadata  
├── prediction  
└── sft_data  
```
The `prediction` folder stores the prediction of three ToolGrad models on the test set. You can run the following command to perform evaluation with LLM judges:
```
python src/eval.py --pred_model toolgrad-1b --dataset ~/YOUR_DATASET_STORAGE_DIR/ToolGrad-5k/
```
You should see the following messages in CMD.
```
judge model: gpt-4.1
100%|████████████████████████████████████████████████████████████████████████████| 500/500 [00:00<00:00, 1384.60it/s]
               Recall  Success Rate     QoR
Model                                      
toolgrad-1b  0.987917      0.955482  93.702
```
This is an exact reproduction of our results.

If you wish to run the LLM judge again, run the following command (note this introduces costs on your OpenAI API):
```
python src/eval.py --pred_model toolgrad-1b \
  --dataset ~/YOUR_DATASET_STORAGE_DIR/ToolGrad-5k/ \
  --overwrite \
  --num_process 16 
```
You should be able to see a new result with similar values of ours. Note that you can adjust the `num_process` dependent on your OpenAI API RPM.


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