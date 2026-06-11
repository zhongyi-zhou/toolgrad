import os
import random
import argparse
import logging
import toolgrad as tog
import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def create_args():
  parser = argparse.ArgumentParser(description="Split SFT data into train/test and combine into JSONL.")
  parser.add_argument(
      '--input_dir',
      type=str,
      required=True,
      help='Input directory containing JSON files.',
  )
  parser.add_argument(
      '--output_dir',
      type=str,
      required=True,
      help='Output directory for split lists and JSONL files.',
  )
  parser.add_argument(
      '--test_ratio',
      type=float,
      default=0.0,
      help='Ratio of data to use for testing (0.0-1.0). Default 0.',
  )
  parser.add_argument(
      '--seed',
      type=int,
      default=42,
      help='Random seed.',
  )
  args = parser.parse_args()
  return args

def save_list_to_txt(data_list: list[str], filepath: str):
  with open(filepath, 'w') as f:
    for item in data_list:
      f.write(item + "\n")

def combine_jsons_to_jsonl(
    json_basenames: list[str],
    input_dirs: list[str],
    output_path: str,
    extra_files: list[str] = None 
):
  data_list = []
  logging.info(f"Combining {len(json_basenames)} records from main pools + {len(extra_files) if extra_files else 0} extra files into {output_path}...")
  
  # 1. Standard merging from basenames + input_dirs
  for basename in tqdm.tqdm(json_basenames, desc="Processing Main Data"):
    for input_dir in input_dirs:
      json_path = os.path.join(input_dir, f"{basename}.json")
      if os.path.exists(json_path):
        try:
          data = tog.utils.data.read_json(json_path)
          data_list.append(data)
        except Exception as e:
          logging.warning(f"Failed to read {json_path}: {e}")

  # 2. Append ALL files from extra_files (Negative Samples)
  if extra_files:
      logging.info(f"Appending {len(extra_files)} extra records...")
      for fname in tqdm.tqdm(extra_files, desc=f"Processing Extra Files"):
        try:
          data = tog.utils.data.read_json(fname)
          data_list.append(data)
        except Exception as e:
           logging.warning(f"Failed to read extra file {fname}: {e}")
  
  tog.utils.data.save_jsonl(data_list, output_path)
  logging.info(f"Saved {len(data_list)} total records to {output_path}")

def split_and_combine_main(
    input_dirs: list[str],
    output_dir: str,
    test_ratio: float,
    seed: int,
    negative_dirs: list[str] = None
):
  random.seed(seed)
  
  if not input_dirs:
      raise ValueError("input_dirs cannot be empty")

  # Use the first directory to determine the split (assuming symmetry)
  primary_input_dir = input_dirs[0]
  if not os.path.exists(primary_input_dir):
    raise FileNotFoundError(f"Directory not found: {primary_input_dir}")
    
  os.makedirs(output_dir, exist_ok=True)
  
  train_path = os.path.join(output_dir, "train.txt")
  test_path = os.path.join(output_dir, "test.txt")
  
  if os.path.exists(train_path) or os.path.exists(test_path):
      logging.warning(f"{train_path} or {test_path} already exists. Proceeding will overwrite them.")

  logging.info(f"Scanning {primary_input_dir} for JSON files...")
  json_files = [f for f in os.listdir(primary_input_dir) if f.endswith('.json')]
  basenames = [os.path.splitext(f)[0] for f in json_files]
  
  if not basenames:
      logging.warning("No JSON files found!")
      return

  random.shuffle(basenames)
  
  # Handle test_ratio logic
  if not (0.0 <= test_ratio <= 1.0):
      logging.warning(f"test_ratio {test_ratio} is out of bounds (0.0-1.0). Clamping.")
      test_ratio = max(0.0, min(1.0, test_ratio))
      
  num_test_count = int(len(basenames) * test_ratio)

  test_list = basenames[:num_test_count]
  train_list = basenames[num_test_count:]
  
  # Handle Negative Samples Split
  train_neg_files = []
  test_neg_files = []
  
  if negative_dirs:
    all_neg_files = []
    for nd in negative_dirs:
      if os.path.exists(nd):
        files = [os.path.join(nd, f) for f in os.listdir(nd) if f.endswith('.json')]
        all_neg_files.extend(files)
      else:
        logging.warning(f"Negative dir {nd} not found.")
    
    random.shuffle(all_neg_files)
    num_neg_test = int(len(all_neg_files) * test_ratio)
    test_neg_files = all_neg_files[:num_neg_test]
    train_neg_files = all_neg_files[num_neg_test:]
    logging.info(f"Split negatives: {len(train_neg_files)} train, {len(test_neg_files)} test (Total: {len(all_neg_files)})")

  logging.info(f"Split data: {len(train_list)} train, {len(test_list)} test (Total Pos: {len(basenames)})")

  # Save the training basenames to 'train.txt'
  save_list_to_txt(train_list, train_path)
  logging.info(f"Saved train list to {train_path}")

  # Save the testing basenames to 'test.txt'
  save_list_to_txt(test_list, test_path)
  logging.info(f"Saved test list to {test_path}")
  
  # Combine to JSONL
  # Train gets the main train list + train negatives
  train_jsonl_path = os.path.join(output_dir, "train.jsonl")
  combine_jsons_to_jsonl(train_list, input_dirs, train_jsonl_path, extra_files=train_neg_files)
  
  # Test gets the main test list + test negatives
  test_jsonl_path = os.path.join(output_dir, "test.jsonl")
  combine_jsons_to_jsonl(test_list, input_dirs, test_jsonl_path, extra_files=test_neg_files)
  
  logging.info("Done!")


if __name__ == "__main__":
  args = create_args()
  
  split_and_combine_main(
      input_dirs=[args.input_dir], # Backward compatibility for CLI
      output_dir=args.output_dir,
      test_ratio=args.test_ratio,
      seed=args.seed,
      negative_dirs=None 
  )
