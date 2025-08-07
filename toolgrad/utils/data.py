import json
import random
import string
from typing import Union
import logging

__all__ = [
    "save_json",
    "read_json",
    "save_jsonl",
    "read_jsonl",
    "get_random_unique_id",
    "replace_keyword",
    "format_int_to_k_digits",
    "json_prefix_string_to_dict",
]


def save_json(data: dict, filepath: str):
  """Save a dictionary to a JSON file."""
  with open(filepath, 'w') as file:
    json.dump(data, file)


def read_json(filepath) -> dict:
  """Load a JSON file and return a dictionary."""
  with open(filepath, 'r') as file:
    data = json.load(file)
  return data


def save_jsonl(jsonl, filepath):
  """Save a list of dictionaries to a JSONL file."""
  with open(filepath, 'w') as f:
    for item in jsonl:
      f.write(json.dumps(item) + '\n')


def read_jsonl(filepath):
  """Load a JSONL file and return a list of dictionaries."""
  with open(filepath, 'r') as file:
    data = [json.loads(line) for line in file]
  return data


def get_random_unique_id(
    length: int = 6,
    existing_ids: list[str] | None = None,
) -> str:
  """Generate a random unique ID."""

  def generate_id():
    return ''.join(
        random.choices(string.ascii_letters + string.digits, k=length))

  max_attempts = 1000
  count = 0
  while True:
    count += 1
    if count > max_attempts:
      raise ValueError("Failed to generate a unique ID.")
    new_id = generate_id()
    if existing_ids is None or new_id not in existing_ids:
      return new_id


def replace_keyword(
    data: Union[dict, str],
    old: str,
    new: str,
) -> Union[dict, str]:
  # If it's a dictionary, iterate over its items
  if isinstance(data, dict):
    new_dict = {}
    for key, value in data.items():
      # Replace keyword in the key if it's a string
      new_key = key.replace(old, new) if isinstance(key, str) else key
      # Recursively process the value
      new_dict[new_key] = replace_keyword(value, old, new)
    return new_dict
  # If it's a string, perform the replacement
  elif isinstance(data, str):
    return data.replace(old, new)

  else:
    raise TypeError(
        f"Unsupported data type: {type(data)}. Only dict and str are supported."
    )


def format_int_to_k_digits(num: int, k: int) -> str:
  """Format an integer to a k-digit string with leading zeros."""
  return f"{num:0{k}d}"


def json_prefix_string_to_dict(json_str_input: str) -> dict:
  """Convert a JSON string with a prefix to a dictionary.

  For example, it converts json from {"key": "value"} to:
  ```json
  {
    "key": "value"
  }
  ```
  """
  json_string = json_str_input.replace("```json", "").replace("```", "").strip()

  json_dict = {}
  try:
    json_dict = json.loads(json_string)

  except json.JSONDecodeError as e:
    logging.error(f"Error decoding JSON: {e}")
    logging.error(f"Problematic JSON string:\n{json_string}")
  return json_dict
