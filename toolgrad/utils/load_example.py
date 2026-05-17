# toolgrad/utils/load_example.py

import json
from importlib import resources


from toolgrad.utils.toolbench import toolbench_data_utils


def load_data(version: str = toolbench_data_utils.DEFAULT_VERSION):
  pkg = f"toolgrad.data.toolbench.{version}"
  with resources.open_text(pkg, "data.json") as fp:
    data = json.load(fp)
  return data


if __name__ == "__main__":
  cfg = load_filtered_v1()
  print("Loaded JSON keys:", list(cfg.keys())[:10])
