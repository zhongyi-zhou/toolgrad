"""Main data formatter

Example:
python src/data/data_format_main.py --workspace_dir $workspace_dir \
    --num_tools 10 --test_ratio 0 --negative_ratio 0.2 \
    --output_format python
"""

import argparse
import sys
import os
import glob
import random
import logging
sys.path.append(os.getcwd())
from src.data import data_format_lib
from src.data import split_lib
from src.data.negative_sample_lib import generate_negatives_main

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def create_args():
  parser = argparse.ArgumentParser(description="Full data processing pipeline: Format -> Split -> Combine.")
  parser.add_argument('--workspace_dir', type=str, required=True, help='Root workspace directory.')
  parser.add_argument('--raw_data_dir', type=str, default=None, help='Input directory of raw workflow JSONs. Defaults to $workspace_dir/data.')
  parser.add_argument('--output_dir', type=str, default=None, help='Base output directory. Defaults to $workspace_dir.')
  parser.add_argument('--num_tools', type=int, default=10, help='Number of tool candidates per sample. Use -1 for all tools.')
  parser.add_argument('--test_ratio', type=float, default=0.0, help='Ratio of data for testing (0.0-1.0).')
  parser.add_argument('--negative_ratio', type=float, default=0.2, help='Ratio of negative samples to generate (0-1).')
  parser.add_argument('--num_processes', type=int, default=16, help='Number of parallel processes.')
  parser.add_argument('--seed', type=int, default=42, help='Random seed.')
  parser.add_argument('--max_description_length', type=int, default=256, help='Maximum tool description length.')
  parser.add_argument('--output_format', type=str, nargs='+', default=['python'], choices=['python'], help='Output format for the AI response.')
  parser.add_argument('--negative_dirs', type=str, nargs='+', default=None, help='Extra directories of negative samples to merge into training set.')
  parser.add_argument('--in_place_neg', action='store_true', help='Sample negatives from raw data and exclude them from positives.')
  return parser.parse_args()


def main():
  args = create_args()

  raw_data_dir = args.raw_data_dir or os.path.join(args.workspace_dir, "data")
  output_dir = args.output_dir or args.workspace_dir
  neg_data_dirs = list(args.negative_dirs) if args.negative_dirs else []
  pos_files = None
  neg_files = None

  # Optional: split raw files into disjoint pos/neg sets upfront
  if args.in_place_neg:
    all_json_files = glob.glob(os.path.join(raw_data_dir, "*.json"))
    random.seed(args.seed)
    random.shuffle(all_json_files)
    target_count = int(len(all_json_files) * args.negative_ratio)
    neg_files = all_json_files[:target_count]
    pos_files = all_json_files[target_count:]
    logging.info(f"In-place split: {len(neg_files)} negative, {len(pos_files)} positive files.")

  # Step 0: Generate negative samples
  if args.negative_ratio > 0:
    logging.info(f"Step 0: Generating negative samples (ratio={args.negative_ratio})...")
    gen_neg_dir = os.path.join(output_dir, "sft_data_neg")
    generate_negatives_main(
        input_dir=raw_data_dir,
        output_dir=gen_neg_dir,
        ratio=args.negative_ratio,
        num_tools=args.num_tools,
        output_format=args.output_format[0],
        seed=args.seed,
        num_processes=args.num_processes,
        file_list=neg_files,
    )
    if gen_neg_dir not in neg_data_dirs:
      neg_data_dirs.append(gen_neg_dir)

  # Step 1: Format positive samples
  sft_data_dirs = []
  for fmt in args.output_format:
    logging.info(f"Step 1 ({fmt}): Formatting raw data to SFT format...")
    current_sft_dir = os.path.join(output_dir, f"sft_data_{fmt}")
    if os.path.exists(current_sft_dir):
      logging.warning(f"SFT data directory already exists: {current_sft_dir}. Overwriting/Adding to it.")
    data_format_lib.format_data_main(
        input_dir=raw_data_dir,
        output_dir=current_sft_dir,
        num_tools=args.num_tools,
        num_processes=args.num_processes,
        seed=args.seed,
        max_description_length=args.max_description_length,
        output_format=fmt,
        file_list=pos_files,
    )
    sft_data_dirs.append(current_sft_dir)

  # Step 2: Split and combine
  logging.info("Step 2: Splitting and combining data...")
  split_lib.split_and_combine_main(
      input_dirs=sft_data_dirs,
      output_dir=output_dir,
      test_ratio=args.test_ratio,
      seed=args.seed,
      negative_dirs=neg_data_dirs,
  )

  logging.info("Pipeline completed successfully!")


if __name__ == '__main__':
  main()
