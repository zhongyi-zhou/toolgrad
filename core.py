import gzip
import json
from importlib import resources
from pathlib import Path
from typing import Iterator, Union


def _list_data_files(group_name: str) -> Iterator[Path]:
  pkg_folder = resources.files("toolgrad") / "data" / group_name

  if not pkg_folder.is_dir():
    raise FileNotFoundError(
        f"Config group '{group_name}' not found in package data.")

  for child in pkg_folder.iterdir():
    assert isinstance(child, Path)
    if child.is_file() and child.suffix in {
        ".json", ".jsonl"
    } or child.name.endswith(".jsonl.gz"):
      yield child


def load_jsonl(path: Union[Path, str]) -> Iterator[dict]:
  path = Path(path)
  if path.suffix == ".gz":
    with gzip.open(path, "rt", encoding="utf-8") as f:
      for line in f:
        yield json.loads(line)
  else:
    with open(path, "r", encoding="utf-8") as f:
      for line in f:
        yield json.loads(line)


def load_json(path: Union[Path, str]) -> dict:
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def load_group(group_name: str) -> Iterator[Union[dict, Iterator[dict]]]:
  """Load a group of JSON/JSONL files.

  For a given group_name (folder under toolgrad/data/), iterate through each file
  and yield either:
    - a dict (if file ends with .json)
    - an iterator over dicts (if file ends with .jsonl or .jsonl.gz)

  Example usage:
    for item in load_group("group_a"):
      if isinstance(item, dict):
        handle_single_json(item)
      else:
        for rec in item:
          handle_each_jsonl_record(rec)
  """

  for data_path in _list_data_files(group_name):
    if data_path.suffix in {".jsonl", ".gz"}:
      yield load_jsonl(data_path)

    elif data_path.suffix == ".json":
      yield load_json(data_path)

    else:
      continue
