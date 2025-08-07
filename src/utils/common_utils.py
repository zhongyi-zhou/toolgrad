import datetime
import json
import random
import string
from typing import Dict, List, Optional, Union


def read_list_from_txt(filepath: str) -> List[str]:
  """Read a list of strings from a text file."""
  with open(filepath, 'r') as f:
    mylist = [line.strip() for line in f if line.strip()]
  return mylist


def format_int_to_k_digits(num: int, k: int) -> str:
  """Format an integer to a k-digit string with leading zeros."""
  return f"{num:0{k}d}"


def get_formatted_date_time(with_year: bool = False) -> str:
  """Get the current date and time formatted as MMDDHHMM."""
  now = datetime.datetime.now(_JST)
  if with_year:
    return now.strftime("%Y%m%d_%H%M")
  else:
    return now.strftime("%m%d_%H%M")


def format_number_with_k_digit(number, k):
  """Format a number into a k-digit string with leading zeros."""
  return f"{number:0{k}d}"


def read_json(filepath) -> Dict:
  """Load a JSON file and return a dictionary."""
  with open(filepath, 'r') as file:
    data = json.load(file)
  return data


def save_json(data: dict, filepath: str):
  """Save a dictionary to a JSON file."""
  with open(filepath, 'w') as file:
    json.dump(data, file)


def save_jsonl(jsonl, filepath):
  with open(filepath, 'w') as f:
    for item in jsonl:
      f.write(json.dumps(item) + '\n')


def read_jsonl(filepath):
  """Load a JSONL file and return a list of dictionaries."""
  with open(filepath, 'r') as file:
    data = [json.loads(line) for line in file]
  return data


def save_txt(data, filepath):
  """Save a dictionary to a JSON file."""
  with open(filepath, 'w') as file:
    file.write(data)


def get_random_unique_id(length=6, existing_ids: Optional[List[str]] = None):
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
