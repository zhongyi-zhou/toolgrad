# toolgrad/utils/load_example.py

import json
from importlib import resources


def load_filtered_v1():
  with resources.open_text("toolgrad.data.toolbench", "filtered_v1.json") as fp:
    data = json.load(fp)
  return data


if __name__ == "__main__":
  cfg = load_filtered_v1()
  print("Loaded JSON keys:", list(cfg.keys())[:10])
